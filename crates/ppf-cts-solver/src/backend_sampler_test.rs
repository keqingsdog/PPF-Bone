//! Actual authored no-root input, backend preparation and CPU device readback.
use super::*;
use clap::Parser;

#[test]
#[ignore = "explicit private authored no-root animated scene; run alone"]
fn no_root_authored_sampler_reprepares_and_uploads_real_fields() {
    let path=std::env::var("PPF_PRIVATE_NO_ROOT_ANIM_SCENE").expect("private authored scene");
    let output=format!("{path}/output");
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let mut scene=Scene::new(&args);
    assert!(scene.joint_input.is_none());assert!(!scene.build_collision_window_table().has_windows);
    let dataset=crate::fresh_authority_dataset(&mut scene);
    let param=crate::builder::make_param(&scene.args());
    let mesh=scene.make_mesh();
    let (times,frames)=scene.rest_shape_schedule().expect("authored rest animation");
    assert!(times.len()>=3 && scene.has_material_animation());
    let rest:Vec<RestShapeKeyframe>=times.into_iter().zip(frames.iter()).map(|(t,frame)|{
        let (i2,i3,ef,et)=crate::builder::compute_inv_rest(frame,&mesh,
            dataset.prop.face.as_slice(),dataset.prop.tet.as_slice(),
            dataset.param_arrays.face.as_slice(),dataset.param_arrays.tet.as_slice(),true);
        (t,i2,i3,ef,et)
    }).collect();
    let area=crate::triutils::face_areas(&mesh.vertex,&mesh.mesh.mesh.face);
    let volume=crate::triutils::tet_volumes(&mesh.vertex,&mesh.mesh.mesh.tet);
    let material:Vec<MaterialKeyframe>=scene.param_anim_times().iter().map(|&t|{
        let props=scene.make_props_at(t,&mesh,&area,&volume);
        let (vertex,edge,hinge)=crate::builder::rederive_animated_tables(&mesh,
            dataset.prop.face.as_slice(),&props.face_params,dataset.prop.edge.as_slice(),
            dataset.prop.hinge.as_slice(),dataset.prop.vertex.as_slice(),
            dataset.param_arrays.vertex.as_slice(),dataset.param_arrays.edge.as_slice(),&area);
        MaterialKeyframe{time:t,face:props.face_params,vertex,edge,hinge}
    }).collect();
    assert!(unsafe{crate::driver::initialize(&dataset,&param)});
    crate::driver::host_seed_capture().unwrap();
    let sample=|h:f32, playback:f32| {
        crate::driver::host_seed_restore().unwrap();
        PRIVATE_PREPARED_OBSERVATIONS.with(|items|items.borrow_mut().clear());
        PRIVATE_PREPARED_CAPTURE.with(|on|on.set(true));
        let result=prepare_attempt_inputs(&scene,0.04,h,playback,None,h*playback,param.prev_dt,false,true,false,false,&rest,Some(&material));
        PRIVATE_PREPARED_CAPTURE.with(|on|on.set(false));result.unwrap();
        let observed=PRIVATE_PREPARED_OBSERVATIONS.with(|items|items.borrow_mut().drain(..).collect::<Vec<_>>());
        let device=crate::driver::private_sample_prepare_device().unwrap();
        (observed,device)
    };
    let direct=sample(0.005,1.0);let long=sample(0.02,1.0);let repeat=sample(0.005,1.0);
    let fast_physical=sample(0.005,2.0);let slow_physical=sample(0.02,0.5);
    crate::driver::host_seed_discard().unwrap();
    let payload=|obs:&[PrivatePreparedObservation],kind|obs.iter().filter(|o|o.kind==kind && o.uploaded).map(|o|o.payload.clone()).collect::<Vec<_>>();
    for kind in ["constraint_fix","rest","material","velocity","angular"] {
        assert_eq!(payload(&direct.0,kind),payload(&repeat.0,kind),"repeat {kind}");
    }
    assert!(!payload(&direct.0,"rest").is_empty() && !payload(&direct.0,"material").is_empty());
    assert_ne!(payload(&direct.0,"rest"),payload(&long.0,"rest"));
    assert_ne!(payload(&direct.0,"material"),payload(&long.0,"material"));
    assert!(payload(&direct.0,"velocity").is_empty() && !payload(&long.0,"velocity").is_empty());
    assert!(payload(&direct.0,"angular").is_empty() && !payload(&long.0,"angular").is_empty());
    for kind in ["velocity", "angular"] {
        assert!(payload(&fast_physical.0,kind).is_empty(), "physical dt must not trigger a future {kind} event");
        assert!(!payload(&slow_physical.0,kind).is_empty(), "scene dt must include crossed {kind} event");
    }
    assert_eq!(direct.1,repeat.1);
    assert!(direct.1.0!=long.1.0 || direct.1.1!=long.1.1,"fixture must change at least one rest channel");
    assert_ne!(direct.1.2,long.1.2);assert_ne!(direct.1.3,long.1.3);
    type Rest=(Vec<Mat2x2f>,Vec<Mat3x3f>,Vec<u8>,Vec<u8>);
    type Material=(Vec<FaceParam>,Vec<VertexParam>,Vec<EdgeParam>,Vec<HingeParam>);
    for sample in [&direct,&long,&repeat] {
        let rest:Rest=bincode::deserialize(&payload(&sample.0,"rest")[0]).unwrap();
        let face:Material=bincode::deserialize(&payload(&sample.0,"material")[0]).unwrap();
        let rest2:Vec<u32>=rest.0.iter().flat_map(|m|m.as_slice().iter().map(|v|v.to_bits())).collect();
        let rest3:Vec<u32>=rest.1.iter().flat_map(|m|m.as_slice().iter().map(|v|v.to_bits())).collect();
        let faces:Vec<(u32,u32)>=face.0.iter().map(|p|(p.mu.to_bits(),p.lambda.to_bits())).collect();
        assert_eq!(rest2,sample.1.0);assert_eq!(rest3,sample.1.1);assert_eq!(faces,sample.1.2);
    }
    eprintln!("PRIVATE_NO_ROOT_SAMPLE rest2_changed={} rest3_changed={} direct_short=0.045 long=0.06 event=0.05",direct.1.0!=long.1.0,direct.1.1!=long.1.1);
}
