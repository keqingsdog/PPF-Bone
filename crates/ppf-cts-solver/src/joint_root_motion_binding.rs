//! Adapter between the existing sealed inventory/native frame and root wire.
//! Intentionally has no CUDA or legacy ABI dependency.
use crate::joint_root_motion_wire::*;

pub const ROOT_MOTION_FILE: &str="bin/pdrd_root_motion_v1.cbor";
fn fail(code:&'static str,field:&str,detail:&str)->RootMotionError {
    RootMotionError {code,body_uuid:None,key_index:None,field:field.into(),detail:detail.into()}
}
pub struct SealedEntry<'a> {pub relative_path:&'a str,pub byte_length:u64,pub sha256:&'a str}

/// Called even when the joint file is absent, so an orphan sidecar is refused.
/// The host already verifies the ENTIRE closed inventory and joint-context
/// digest. This helper additionally binds these exact bytes to their root entry.
pub fn decode_bound(required_features:&[String], sidecar:Option<&[u8]>,
                    entry:Option<SealedEntry<'_>>, scene_uuid:&str)->RootResult<Vec<ValidatedRootMotion>> {
    let features=required_features.iter().filter(|f|f.as_str()==ROOT_FEATURE || f.as_str()==ROOT_FEATURE_V2 || f.as_str()==ROOT_FEATURE_V3).collect::<Vec<_>>();
    if features.len()>1 {return Err(fail(CODE_REQUIRED_FEATURES,"required_features","exactly one root version feature allowed"));}
    let body_controls=required_features.iter().any(|f|f==BODY_CONTROLS_FEATURE);
    if body_controls && (features.len()!=1 || !matches!(features[0].as_str(),ROOT_FEATURE_V2|ROOT_FEATURE_V3)) {
        return Err(fail(CODE_REQUIRED_FEATURES,"required_features","body controls require root-v2 or root-v3 sidecar feature"));
    }
    let feature=features.first().map(|f|f.as_str());
    let version=sidecar.map(|bytes|peek_sidecar_version(bytes).map(|v|v.0)).transpose()?;
    validate_sidecar_version_obligation_v9(feature,sidecar.is_some(),version)?;
    let Some(bytes)=sidecar else {
        if entry.is_some() {return Err(fail(CODE_INPUT_HASH_MISMATCH,ROOT_MOTION_FILE,"inventory claims absent sidecar"));}
        return Ok(Vec::new());
    };
    let entry=entry.ok_or_else(||fail(CODE_INPUT_HASH_MISMATCH,ROOT_MOTION_FILE,"sidecar not in sealed inventory"))?;
    if entry.relative_path!=ROOT_MOTION_FILE {return Err(fail(CODE_INPUT_HASH_MISMATCH,ROOT_MOTION_FILE,"wrong inventory path"));}
    verify_sidecar_snapshot(bytes,entry.byte_length,entry.sha256)?;
    let roots=match version {
        Some(1)=>vec![decode_sidecar(bytes)?],
        Some(2)=>decode_sidecar_v2(bytes)?.bodies,
        Some(3)=>decode_sidecar_v3(bytes)?.bodies,
        _=>return Err(fail(CODE_VERSION_OR_KIND,ROOT_MOTION_FILE,"unsupported root version")),
    };
    if !body_controls && roots.iter().any(|r|r.mode==RootMotionMode::Dynamic || r.control_events.iter().any(|e|e.pose_binding.is_some())) {
        return Err(fail(CODE_REQUIRED_FEATURES,"required_features","new control modes/policies require pdrd_body_controls_v1"));
    }
    if roots.iter().any(|r|r.scene_uuid!=scene_uuid) {
        return Err(fail(CODE_BODY_BINDING,"scene_uuid","sidecar/context scene mismatch"));
    }
    if roots.iter().map(|r|r.keys.len()).sum::<usize>()>MAX_KEY_RECORDS_TOTAL_V2
        || roots.iter().any(|r|r.keys.len()>MAX_KEYS_PER_ROOT_V2) {
        return Err(fail(CODE_CAPACITY_OVERFLOW,"keys","root key capacity exceeded"));
    }
    let mut uuids=std::collections::HashSet::new();
    if roots.iter().any(|r|!uuids.insert(&r.body_uuid)) {
        return Err(fail(CODE_ROOT_BATCH,"body_uuid","duplicate root UUID"));
    }
    Ok(roots)
}

/// New native support gates. Hard-joint pairing remains an independent gate:
/// don't loosen the hard-D6/hinge scope as a side effect of accepting roots.
pub struct NativeCapabilityEvidence {
    pub playback: f32, pub world_scale: f32,
    /// Every consumed primitive/analytic friction input was validated from
    /// the retained sealed generation. A scalar zero is not that evidence.
    pub friction_domain_validated: bool,
    pub root_module_installed: bool,
    pub key_cut_installed: bool, pub projected_solve_installed: bool,
    pub endpoint_history_gate_installed: bool,
    pub curved_ccd_installed: bool, pub control_boundary_installed: bool,
}
pub fn gate_native(roots:&[ValidatedRootMotion], e:&NativeCapabilityEvidence)->RootResult<()> {
    if roots.is_empty() {return Ok(());}
    if e.playback!=1.0 || e.world_scale!=1.0 || !e.friction_domain_validated
        || !e.root_module_installed || !e.key_cut_installed
        || !e.projected_solve_installed || !e.endpoint_history_gate_installed {
        return Err(fail(CODE_NATIVE_CAPABILITY,"neutral_root","root native slice requires integrated seams, playback=1, world_scale=1 and validated finite nonnegative upstream friction"));
    }
    if roots.iter().any(|r|r.keys.iter().any(|k|k.pose.rotation_world_wxyz!=r.keys[0].pose.rotation_world_wxyz)) && !e.curved_ccd_installed {
        return Err(fail(CODE_NATIVE_CAPABILITY,"curved_ccd","curved trajectory envelope, subdivision and native CCD seams required"));
    }
    if roots.iter().any(|r|!r.control_events.is_empty()) && !e.control_boundary_installed {
        return Err(fail(CODE_NATIVE_CAPABILITY,"control_events","native physical-boundary readback and atomic mode switch required"));
    }
    Ok(())
}

/// Validate all roots, membership-disjointness and overlap with EVERY scheduled
/// old fix/pull pin, not just currently active pins. Rest, initial and identities
/// are obtained from the same consumed native scene snapshot.
pub fn bind_native_frames(roots:&[ValidatedRootMotion], evidence:&[NativeFrameEvidence<'_>],
                          scheduled_pin_vertices:&[u32])->RootResult<Vec<f64>> {
    bind_native_frames_with_body_controls(roots,evidence,scheduled_pin_vertices,false)
}
pub fn bind_native_frames_with_body_controls(roots:&[ValidatedRootMotion],evidence:&[NativeFrameEvidence<'_>],
    scheduled_pin_vertices:&[u32],body_controls:bool)->RootResult<Vec<f64>> {
    if roots.len()!=evidence.len() {return Err(fail(CODE_BODY_BINDING,"native_frame","one evidence row per root required"));}
    let pins:std::collections::HashSet<_>=scheduled_pin_vertices.iter().copied().collect();
    let mut membership=std::collections::HashSet::new();
    let mut residuals=Vec::new();
    for (root,e) in roots.iter().zip(evidence) {
        if e.vertex_indices.iter().any(|v|pins.contains(v)) {
            return Err(fail(CODE_OLD_PIN_OVERLAP,"constraint.fix/pull","root overlaps scheduled legacy pin"));
        }
        if e.vertex_indices.iter().any(|v|!membership.insert(*v)) {
            return Err(fail(CODE_BODY_BINDING,"vertex_membership","root bodies overlap vertices"));
        }
        residuals.push(root.validate_initial_native_frame_with_body_controls(e,body_controls)?);
    }
    Ok(residuals)
}

#[cfg(test)]
mod friction_capability_tests {
    use super::*;
    use sha2::Digest;
    fn root_v3_bytes()->Vec<u8> {
        let wf=|v:f64|WireF64(v);
        let body=RootMotionBodyWireV3{body_uuid:"22222222-2222-2222-2222-222222222222".into(),runtime_body_index:0,mode:RootMotionMode::Kinematic,
            keys:vec![RootMotionKeyWireV1{time_s:wf(0.0),center_world_m:[wf(0.0);3],rotation_world_wxyz:[wf(1.0),wf(0.0),wf(0.0),wf(0.0)]},
                RootMotionKeyWireV1{time_s:wf(1.0),center_world_m:[wf(1.0),wf(0.0),wf(0.0)],rotation_world_wxyz:[wf(1.0),wf(0.0),wf(0.0),wf(0.0)]}],control_events:vec![],
            reference_transform:RootReferenceTransformWireV3{translation_local_m:[wf(0.0);3],rotation_local_wxyz:[wf(1.0),wf(0.0),wf(0.0),wf(0.0)]}};
        let wire=RootMotionEnvelopeWireV3{version:3,kind:ROOT_KIND.into(),payload:RootMotionPayloadWireV3{scene_uuid:"11111111-1111-1111-1111-111111111111".into(),
            interpolation:INTERPOLATION.into(),extrapolation:EXTRAPOLATION.into(),bodies:vec![body]}};
        let mut bytes=Vec::new();ciborium::ser::into_writer(&wire,&mut bytes).unwrap();bytes
    }
    fn root_v2_bytes()->Vec<u8> {
        let wf=|v:f64|WireF64(v);
        let body=RootMotionBodyWireV1{body_uuid:"22222222-2222-2222-2222-222222222222".into(),runtime_body_index:0,mode:RootMotionMode::Kinematic,
            keys:vec![RootMotionKeyWireV1{time_s:wf(0.0),center_world_m:[wf(0.0);3],rotation_world_wxyz:[wf(1.0),wf(0.0),wf(0.0),wf(0.0)]},
                RootMotionKeyWireV1{time_s:wf(1.0),center_world_m:[wf(1.0),wf(0.0),wf(0.0)],rotation_world_wxyz:[wf(1.0),wf(0.0),wf(0.0),wf(0.0)]}],control_events:vec![]};
        let wire=RootMotionEnvelopeWireV1{version:2,kind:ROOT_KIND.into(),payload:RootMotionPayloadWireV1{scene_uuid:"11111111-1111-1111-1111-111111111111".into(),
            interpolation:INTERPOLATION.into(),extrapolation:EXTRAPOLATION.into(),bodies:vec![body]}};
        let mut bytes=Vec::new();ciborium::ser::into_writer(&wire,&mut bytes).unwrap();bytes
    }
    fn root_v1_bytes()->Vec<u8> {
        let wf=|v:f64|WireF64(v);
        let body=RootMotionBodyWireV1{body_uuid:"22222222-2222-2222-2222-222222222222".into(),runtime_body_index:0,mode:RootMotionMode::Static,
            keys:vec![RootMotionKeyWireV1{time_s:wf(0.0),center_world_m:[wf(0.0);3],rotation_world_wxyz:[wf(1.0),wf(0.0),wf(0.0),wf(0.0)]}],control_events:vec![]};
        let wire=RootMotionEnvelopeWireV1{version:1,kind:ROOT_KIND.into(),payload:RootMotionPayloadWireV1{scene_uuid:"11111111-1111-1111-1111-111111111111".into(),
            interpolation:INTERPOLATION.into(),extrapolation:EXTRAPOLATION.into(),bodies:vec![body]}};
        let mut bytes=Vec::new();ciborium::ser::into_writer(&wire,&mut bytes).unwrap();bytes
    }
    fn bound(features:&[&str],bytes:Option<&[u8]>)->RootResult<Vec<ValidatedRootMotion>> {
        let entry=bytes.map(|b| {let h=sha2::Sha256::digest(b);let hash=format!("{h:x}");(hash,b.len() as u64)});
        let refs=entry.as_ref().map(|(h,n)|SealedEntry{relative_path:ROOT_MOTION_FILE,byte_length:*n,sha256:h});
        decode_bound(&features.iter().map(|x|x.to_string()).collect::<Vec<_>>(),bytes,refs,"11111111-1111-1111-1111-111111111111")
    }
    #[test]
    fn roots_require_validated_retained_friction_domain() {
        let roots=[ValidatedRootMotion {scene_uuid:"scene".into(),body_uuid:"body".into(),runtime_body_index:0,
            mode:RootMotionMode::Static,keys:vec![NormalizedKey {time_s:0.0,
                pose:RootPose {center_world_m:[0.0;3],rotation_world_wxyz:[1.0,0.0,0.0,0.0]}}],control_events:vec![],reference_transform:None}];
        let mut evidence=NativeCapabilityEvidence {playback:1.0,world_scale:1.0,friction_domain_validated:false,
            root_module_installed:true,key_cut_installed:true,projected_solve_installed:true,
            endpoint_history_gate_installed:true,curved_ccd_installed:true,control_boundary_installed:true};
        assert_eq!(gate_native(&roots,&evidence).unwrap_err().code,CODE_NATIVE_CAPABILITY);
        evidence.friction_domain_validated=true;
        gate_native(&roots,&evidence).unwrap();
        evidence.world_scale=2.0;
        assert_eq!(gate_native(&roots,&evidence).unwrap_err().code,CODE_NATIVE_CAPABILITY);
    }
    #[test]
    fn decode_bound_v3_requires_its_exact_feature_and_control_dependency() {
        let bytes=root_v3_bytes();
        assert_eq!(bound(&[ROOT_FEATURE_V3],Some(&bytes)).unwrap().len(),1);
        assert!(bound(&[ROOT_FEATURE_V3],Some(&root_v2_bytes())).is_err());
        assert!(bound(&[ROOT_FEATURE_V2],Some(&bytes)).is_err());
        assert!(bound(&[],Some(&bytes)).is_err());
        assert!(bound(&[ROOT_FEATURE_V3],None).is_err());
        assert!(bound(&[ROOT_FEATURE, BODY_CONTROLS_FEATURE],Some(&bytes)).is_err());
        assert_eq!(bound(&[ROOT_FEATURE_V3,BODY_CONTROLS_FEATURE],Some(&bytes)).unwrap().len(),1);
        assert_eq!(bound(&[ROOT_FEATURE_V2],Some(&root_v2_bytes())).unwrap().len(),1);
        assert_eq!(bound(&[ROOT_FEATURE],Some(&root_v1_bytes())).unwrap().len(),1);
    }
}
