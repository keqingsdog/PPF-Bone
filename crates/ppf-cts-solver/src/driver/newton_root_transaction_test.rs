//! Private CPU integration fixture. Runs only when explicitly requested; the
//! input scene is supplied by environment, never installed as a build dependency.
use clap::Parser;
use ppf_cts_compute::Device;

#[test]
#[ignore = "requires an explicit private 05 native session"]
fn root_material_newton_cap_restores_real_buffers_and_resamples() {
    super::test_device_fault::start_private_status();
    let path=std::env::var("PPF_PRIVATE_NEWTON_SCENE").expect("explicit private scene path");
    let output=std::env::var("PPF_PRIVATE_NEWTON_OUTPUT").expect("separate private test output path");
    std::fs::create_dir_all(&output).unwrap();
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let require_restoring=args.require_restoring_curves_v1;
    let mut scene=crate::scene::Scene::new(&args);
    let owned=scene.joint_input.as_ref().expect("joint input").records().clone();
    let overlay=crate::twist_overlay::material_spin_from_validated(&owned).unwrap();
    let soft=crate::cross_links::prepare_batch(scene.joint_input.as_ref().unwrap(),scene.args().world_scaling as f64).unwrap();
    let mut data=crate::fresh_authority_dataset(&mut scene);
    if let Some(prepared)=soft.as_ref().and_then(|s|s.connected_collision.as_ref()) {
        crate::connected_collision_native::install(&mut data,prepared,0,std::path::Path::new(&output)).unwrap();
    } else { unsafe {crate::native_joint::filter_connected(&mut data,&owned)}; }
    let mut snhk=std::collections::BTreeMap::<u32,std::collections::BTreeSet<usize>>::new();
    for (tet,prop) in data.mesh.mesh.tet.as_slice().iter().zip(data.prop.tet.as_slice()) {
        let material=&data.param_arrays.tet.as_slice()[prop.param_index as usize];
        if material.model==crate::data::Model::SNHk {
            assert_eq!(material.plasticity,0.0);
            let object=data.prop.vertex.as_slice()[tet[0] as usize].object_index;
            snhk.entry(object).or_default().extend(tet.iter().map(|v|*v as usize));
        }
    }
    assert_eq!(snhk.len(),2,"fixture must contain two ordinary SNHK objects");
    let mut param=crate::builder::make_param(&scene.args());
    assert_eq!(param.playback,1.0);
    assert!(scene.rest_shape_schedule().is_none_or(|(keys,_)|keys.is_empty()));
    assert!(unsafe{super::initialize(&data,&param)});
    super::install_joints_from_rest(&owned,None);
    super::install_hard_joints(&owned.hard_records,owned.payload_version==5,owned.roots.len(),data.translation_lock.size as usize).unwrap();
    if let Some(s)=soft.as_ref(){super::install_soft_joints_from_rest(s,false).unwrap();}
    let curves=crate::restoring_curve::prepare_fresh(std::path::Path::new(&path),require_restoring,0).unwrap()
        .map(|prepared|crate::restoring_native::verify_and_pack(&prepared,&scene,&data,std::path::Path::new(&path)).unwrap());
    let curves=crate::cross_links::merge_curves(scene.joint_input.as_ref(),curves,&scene,Some(&data)).unwrap();
    if let Some(curves)=curves.as_ref() {super::install_restoring(curves,data.constraint.stitch.size as usize).unwrap();}
    super::prepare_material_fit_selections(None,Some(&overlay.mask),None).unwrap();
    let q=super::sample_committed_material_spin(&overlay.mask).unwrap();
    let rows=q.into_iter().zip(&overlay.mask).filter_map(|(mut q,on)|if *on {let n=q.iter().map(|x|x*x).sum::<f64>().sqrt();assert!(n.is_finite() && n>0.99 && n<1.01);for v in &mut q {*v/=n;}Some(crate::material_spin_history::MaterialSpinRow{q_wxyz:q,a:[0.;3]})}else{None}).collect();
    let history=crate::material_spin_history::AcceptedMaterialSpin{scene_uuid:overlay.scene_uuid.clone(),joint_uuids:overlay.selected_joint_uuids.clone(),time:param.time,generation:0,rows};
    super::activate_material_spin(history,&overlay.scene_uuid,&overlay.selected_joint_uuids,param.time,overlay.mask.clone()).unwrap();
    {
        let guard=super::SOLVER.lock().unwrap();let state=guard.as_ref().unwrap();
        let (_,soft_active,drive_count,_)=state.joints.private_swing_scope_facts();
        let root=state.roots.transaction.as_ref();
        eprintln!("PRIVATE_NEWTON_SCOPE playback={} hard={} fem={} root_count={} root_active={} root_rotating={} root_events={} root_single_kinematic={} soft={} drives={} material={} accepted={}",
            param.playback,state.hard.active(),state.fem_roots.runtime.is_some(),owned.roots.len(),root.map_or(0,|r|r.active_indices().len()),state.roots.rotating(),state.roots.has_control_events(),root.is_some_and(|r|r.single_kinematic_retry_root()),soft_active,drive_count,state.joints.material_spin_active(),state.joints.accepted_material_spin().is_some());
    }
    assert!(super::host_newton_root_scope(param.playback).unwrap());
    let advance=|param:&mut crate::data::ParamSet| {
        super::host_seed_capture().unwrap();
        let plan=unsafe{super::prepare_initial_newton_root_plan(param.dt)}.unwrap();
        crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,Some(plan)).unwrap();
        let result=unsafe{super::advance_host(0,16,true,false,Some(plan))}.unwrap();
        super::host_seed_discard().unwrap();result
    };
    assert!(matches!(advance(&mut param),super::HostAdvanceOutcome::Accepted(r) if r.success()),"warmup must produce real accepted history");
    let snapshot=|| {
        let mut guard=super::SOLVER.lock().unwrap();let state=guard.as_mut().unwrap();
        super::with_device(|d| {
            state.positions.download(d)?;state.positions_prev.download(d)?;
            let pos=state.positions.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let prev=state.positions_prev.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let mut rigid=vec![crate::data::PdrdRigidState::default();state.rigid_state.len()];state.rigid_state.read(d,0,&mut rigid)?;
            let rigid=rigid.iter().map(|r|(r.anchor.map(f32::to_bits),r.x.map(f32::to_bits),r.rotation.map(f32::to_bits),r.reference_inertia.map(f32::to_bits),r.mass_total.to_bits(),r.count)).collect::<Vec<_>>();
            let mut running=vec![0f32;state.rigid_running_rotation.len()];state.rigid_running_rotation.read(d,0,&mut running)?;
            let mut omega=vec![0f32;state.grain_omega.len()];state.grain_omega.read(d,0,&mut omega)?;
            let root=state.roots.transaction.as_ref().unwrap().export_checkpoint().unwrap();
            let curves=state.joints.curves.snapshot_trial(d)?;
            let mut fix=vec![crate::data::FixPair::default();state.fix.len()];state.fix.read(d,0,&mut fix)?;
            Ok((pos,prev,rigid,running.into_iter().map(f32::to_bits).collect::<Vec<_>>(),omega.into_iter().map(f32::to_bits).collect::<Vec<_>>(),bincode::serialize(&root).unwrap(),bincode::serialize(&curves).unwrap(),state.joints.accepted_material_spin().cloned(),(state.roots.transaction.as_ref().unwrap().hard_epoch().0,state.roots.transaction.as_ref().unwrap().commit_sequence),state.restoring_commit_count,(bincode::serialize(&fix).unwrap(),state.rigid_rotation_seeded,state.restoring_commit_time)))
        }).unwrap()
    };
    let mut before=snapshot();
    for _ in 0..15 {
        let material_nonzero=before.7.as_ref().unwrap().rows.iter().any(|r|r.a.iter().any(|v|*v!=0.));
        let solids_nonzero=snhk.values().all(|vs|vs.iter().any(|v|(0..3).any(|axis|before.0[3*v+axis]!=before.1[3*v+axis])));
        if material_nonzero && solids_nonzero {break;}
        assert!(matches!(advance(&mut param),super::HostAdvanceOutcome::Accepted(r) if r.success()),"warmup stopped before nonzero accepted history");
        before=snapshot();
    }
    {
        let guard=super::SOLVER.lock().unwrap();let state=guard.as_ref().unwrap();
        let distance=state.joints.curves.binding_metadata.iter().filter(|b|b.kind==crate::cross_links::DISTANCE_KIND).count();
        assert_eq!(distance,96,"fixture must retain all authored cross-link distance bindings");
        let curves=super::with_device(|d|Ok(state.joints.curves.snapshot_trial(d)?)).unwrap().expect("cross-link history");
        assert!(curves.history.iter().any(|h|h.initialized!=0 && (h.q0!=0. || h.d0.iter().any(|v|*v!=0.))),"cross-link history must be nonzero");
    }
    assert!(before.7.as_ref().unwrap().generation>0);
    assert!(before.7.as_ref().unwrap().rows.iter().any(|r|r.a.iter().any(|v|*v != 0.)),"fixture must have nonzero material history");
    for vertices in snhk.values(){assert!(vertices.iter().any(|v|(0..3).any(|axis|before.0[3*v+axis]!=before.1[3*v+axis])),"each SNHK object must have nonzero physical position history");}
    let clock=(param.time.to_bits(),param.time_f32.to_bits(),param.prev_dt.to_bits(),param.dt.to_bits());
    let limit=param.max_newton_steps;param.max_newton_steps=1; // test-only cap; scene remains untouched
    super::roots::PRIVATE_TEST_BOUNDARY_COUNT.store(0,std::sync::atomic::Ordering::SeqCst);
    super::host_seed_capture().unwrap();
    let plan=unsafe{super::prepare_initial_newton_root_plan(param.dt)}.unwrap();
    crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,Some(plan)).unwrap();
    let prepared_before=snapshot();
    let attempt_seq_before=super::SOLVER.lock().unwrap().as_ref().unwrap().roots.transaction.as_ref().unwrap().hard_epoch().1;
    let (token,failed)=match unsafe{super::advance_host(0,16,true,false,Some(plan))}.unwrap() {
        super::HostAdvanceOutcome::NewtonCapRejected{refusal,continuation}=>(continuation,refusal.failed_dt),
        _=>panic!("test cap did not reach the real Newton B0 refusal"),
    };
    assert_ne!(token,0);assert_eq!((param.time.to_bits(),param.time_f32.to_bits(),param.prev_dt.to_bits(),param.dt.to_bits()),clock);
    let attempt_seq_after=super::SOLVER.lock().unwrap().as_ref().unwrap().roots.transaction.as_ref().unwrap().hard_epoch().1;
    assert!(attempt_seq_after>attempt_seq_before,"rejected attempt identity must not be reused");
    let rolled_back=snapshot();assert!(prepared_before==rolled_back,"advance itself must restore prepared buffers and histories");
    assert_eq!(super::SOLVER.lock().unwrap().as_ref().unwrap().joints.soft.private_retry_cache(),(false,false,None),"failed trial must invalidate soft scratch cache");
    super::host_seed_restore().unwrap();
    let after=snapshot();
    assert!(before.0==after.0 && before.1==after.1 && before.2==after.2 && before.3==after.3 && before.4==after.4 && before.5==after.5 && before.6==after.6 && before.7==after.7 && before.8==after.8 && before.9==after.9 && before.10.1==after.10.1 && before.10.2==after.10.2,"accepted buffers and histories must roll back");
    // Fix is a derived host input, re-uploaded for each endpoint, rather than
    // accepted history. The trial rollback retains the prepared input.
    assert!(after.10.0==prepared_before.10.0,"prepared fix input must remain intact");
    {let mut guard=super::SOLVER.lock().unwrap();let state=guard.as_mut().unwrap();assert!(state.roots.view(&state.rigid).is_none(),"rejected root targets must be invalidated");}
    let next=super::adaptive_retry::next_half_h(failed,param.time).unwrap();param.max_newton_steps=limit;param.dt=next;
    super::host_seed_restore().unwrap();
    let plan=unsafe{super::prepare_host_plan(token,next)}.unwrap();
    let observed=crate::backend::private_prepare_03(&scene,param.time,next,param.playback,Some(plan)).unwrap();
    let sampled:(f64,f64,f32)=bincode::deserialize(&observed.iter().find(|x|x.0=="clock").unwrap().2).unwrap();
    assert_eq!(sampled.1.to_bits(),plan.end_s().to_bits());assert_eq!(sampled.2.to_bits(),plan.integration_h().to_bits());
    let uploaded=&observed.iter().find(|x|x.0=="constraint_fix" && x.1).expect("new endpoint fix uploaded").2;
    assert!(snapshot().10.0==*uploaded,"device fix must match the new endpoint upload");
    assert!(matches!(unsafe{super::advance_host(token,15,true,false,Some(plan))}.unwrap(),super::HostAdvanceOutcome::Accepted(r) if r.success()));
    super::host_seed_discard().unwrap();
    {
        let mut guard=super::SOLVER.lock().unwrap();let state=guard.as_mut().unwrap();
        let root=state.roots.transaction.as_ref().unwrap();
        assert_eq!(root.committed_time_s.to_bits(),param.time.to_bits());
        assert!(param.time>sampled.0 && param.time<=plan.end_s(),"actual accepted endpoint must lie within resampled plan");
        let view=state.roots.view(&state.rigid).expect("accepted root targets");
        let mut targets=vec![crate::joint_root_motion_gpu::RootTargetGpu::default();view.count as usize];
        super::with_device(|d|{let bytes=unsafe{std::slice::from_raw_parts_mut(targets.as_mut_ptr().cast::<u8>(),targets.len()*std::mem::size_of::<crate::joint_root_motion_gpu::RootTargetGpu>())};d.read(view.targets,0,bytes)?;Ok(())}).unwrap();
        for actual in targets {
            assert_eq!(actual.enabled,1);
            let pose=state.roots.transaction.as_ref().unwrap().pose_at_committed(actual.body).unwrap();
            let expected=crate::joint_root_motion_gpu::RootTargetGpu::sampled(actual.body,pose,actual.list_start,actual.list_count).unwrap();
            assert_eq!(actual,expected,"device target must match root pose at actual accepted time");
        }
    }
    let cache=super::SOLVER.lock().unwrap().as_ref().unwrap().joints.soft.private_retry_cache();
    assert!(cache.0 && cache.1 && cache.2.is_some_and(|h|h>0. && h<=plan.integration_h()),"retry must refreeze cross cache for its actual new duration");
    let committed=snapshot();assert_eq!(committed.7.unwrap().generation,before.7.unwrap().generation+1);
    assert_eq!(committed.8.1,before.8.1+1);
    assert_eq!(super::roots::PRIVATE_TEST_BOUNDARY_COUNT.load(std::sync::atomic::Ordering::SeqCst),1);
    // Run the actual B3 and PCG, with a controlled CCD fraction and a test-only
    // cap verdict after the real solve. The cap must carry B3's shortened dt.
    super::host_seed_capture().unwrap();
    let b3_start=param.time;let b3_initial=param.dt;let mut b3_calls=0;
    let b3_run=super::adaptive_retry::run_host_attempts(|h,index,token,budget| {
        b3_calls+=1;param.dt=h;
        let plan=if index==0 {unsafe{super::prepare_initial_newton_root_plan(h)}?} else {
            super::host_seed_restore()?;
            assert_eq!(h.to_bits(),(super::step::private_b3_dt()*0.5).to_bits());
            unsafe{super::prepare_host_plan(token,h)}?
        };
        let sampled=crate::backend::private_prepare_03(&scene,param.time,h,param.playback,Some(plan))?;
        let clock:(f64,f64,f32)=bincode::deserialize(&sampled.iter().find(|x|x.0=="clock").unwrap().2).unwrap();
        assert_eq!(clock.2.to_bits(),plan.integration_h().to_bits());
        if index==0 {super::step::arm_private_b3_pcg();}
        let outcome=unsafe{super::advance_host(token,budget,true,false,Some(plan))}?;
        if index==0 {
            let local=super::step::private_b3_dt();assert!(local>0. && local<plan.integration_h());
            match &outcome {super::HostAdvanceOutcome::PcgCap{continuation,failed_dt,..}=>{assert_ne!(*continuation,0);assert_eq!(failed_dt.to_bits(),local.to_bits());},_=>panic!("B3 cap must return typed continuation")}
            let guard=super::SOLVER.lock().unwrap();assert_eq!(guard.as_ref().unwrap().host_retry_pending.as_ref().unwrap().failed_h_bits,local.to_bits());
        }
        Ok(outcome)
    },super::cancel_host_retry,super::host_seed_discard,b3_initial,b3_start,16,b3_initial).unwrap();
    assert_eq!(b3_calls,2);assert_eq!(b3_run.retry_count,1);assert!(b3_run.result.success());
    // The injected post-C5 fault must not open a retry or undo the commit.
    let committed_time=param.time;
    super::host_seed_capture().unwrap();
    let next_plan=unsafe{super::prepare_initial_newton_root_plan(param.dt)}.unwrap();
    crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,Some(next_plan)).unwrap();
    super::step::set_private_test_commit_stage(2);
    let failure=match unsafe{super::advance_host(0,16,true,false,Some(next_plan))} {Err(e)=>e,Ok(_)=>panic!("post-C5 injected fault must remain fatal")};
    assert!(failure.detail.contains("PRIVATE_TEST_POST_C5"));
    assert!(param.time>committed_time);
    assert!(super::SOLVER.lock().unwrap().as_ref().unwrap().host_retry_pending.is_none());
    super::host_seed_discard().unwrap();
}










