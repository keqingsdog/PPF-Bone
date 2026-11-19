//! Versioned runtime sidecar; old positional bincode State stays unchanged.
use std::path::Path;
use bincode::Options;
#[derive(Clone,Debug,PartialEq,serde::Serialize,serde::Deserialize)]
pub(crate) struct History {
    pub kind:String,pub version:u32,pub frame:i32,pub accepted_time:f64,pub previous_dt:f32,
    pub sealed_frames:i32,pub fps:f64,pub param_bytes:Vec<u8>,
    pub scene_uuid:String,pub scene_inputs_sha256:String,
    pub filtered_links:usize,pub joint_accepted:u64,
    pub adaptive:crate::driver::adaptive_controller::Controller,
    pub adaptive_enabled:bool,pub adaptive_start_time:f64,
    pub runtime:crate::driver::checkpoint::Snapshot,
}
const MAX_BYTES:u64=256*1024*1024;
// The legacy History and Snapshot layouts remain frozen for positional bincode.
const V2_MAGIC: &[u8; 8] = b"PPFCHV2\0";
const V3_MAGIC: &[u8; 8] = b"PPFCHV3\0";
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct TwistModeMetadata {
    pub fit_scheme: String,
    pub effective_model: String,
    pub source_models: Vec<String>,
    pub selected_joint_uuids: Vec<String>,
    pub migration_time: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct HistoryV2 {
    pub version: u32,
    pub legacy: History,
    pub twist: crate::continuous_twist::AcceptedTwist,
    pub mode: TwistModeMetadata,
}
impl HistoryV2 {
    pub(crate) fn encode_payload(&self) -> Result<Vec<u8>, String> {
        if self.version != 2 { return Err("CHECKPOINT_HISTORY_INVALID: V2 version".into()); }
        let mut bytes = V2_MAGIC.to_vec();
        bytes.extend(bincode::serialize(self).map_err(|e| format!("CHECKPOINT_HISTORY_INVALID: {e}"))?);
        Ok(bytes)
    }
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum HistoryPayload { V1(History), V2(HistoryV2), V3(HistoryV3) }
/// V3 stores the material state itself; a V1/V2 twist cannot supply its bend path.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct HistoryV3 {
    pub version: u32,
    pub legacy: History,
    pub model: String,
    pub rigid_material: Option<crate::material_spin_history::AcceptedMaterialSpin>,
    pub fem_material: Option<crate::material_spin_history::AcceptedMaterialSpin>,
    pub twist: Option<HistoryV2>,
}
impl HistoryV3 {
    pub(crate) fn validate_material(&self,scene:&str,joints:&[String])->Result<(),String>{
        if self.version!=3 || self.model!=crate::material_spin_history::MODEL || self.legacy.scene_uuid!=scene {
            return Err("CHECKPOINT_MATERIAL_SPIN_INVALID: model or scene".into());
        }
        self.rigid_material.as_ref().ok_or("CHECKPOINT_MATERIAL_SPIN_INVALID: missing rigid history")?.validate(scene,joints,self.legacy.accepted_time).map_err(str::to_owned)
    }
    fn encode_payload(&self)->Result<Vec<u8>,String>{
        if self.version!=3 || self.model!=crate::material_spin_history::MODEL || (self.rigid_material.is_none()&&self.fem_material.is_none()) || self.twist.as_ref().is_some_and(|t|t.version!=2||t.legacy!=self.legacy) {return Err("CHECKPOINT_MATERIAL_SPIN_INVALID: version, model or embedded history".into());}
        let mut bytes=V3_MAGIC.to_vec();
        bytes.extend(bincode::serialize(self).map_err(|e|format!("CHECKPOINT_HISTORY_INVALID: {e}"))?);
        Ok(bytes)
    }
}
// V2 errors never fall back to V1.
pub(crate) fn decode_payload(payload: &[u8]) -> Result<HistoryPayload, String> {
    if let Some(data) = payload.strip_prefix(V3_MAGIC) {
        let value: HistoryV3 = bincode::DefaultOptions::new().with_fixint_encoding().with_limit(MAX_BYTES)
            .reject_trailing_bytes().deserialize(data).map_err(|e| format!("CHECKPOINT_HISTORY_INVALID: {e}"))?;
        if value.version!=3 || value.model!=crate::material_spin_history::MODEL || (value.rigid_material.is_none()&&value.fem_material.is_none()) || value.twist.as_ref().is_some_and(|t|t.version!=2||t.legacy!=value.legacy) {return Err("CHECKPOINT_MATERIAL_SPIN_INVALID: version, model or embedded history".into());}
        Ok(HistoryPayload::V3(value))
    } else if let Some(data) = payload.strip_prefix(V2_MAGIC) {
        let value: HistoryV2 = bincode::DefaultOptions::new().with_fixint_encoding().with_limit(MAX_BYTES)
            .reject_trailing_bytes().deserialize(data).map_err(|e| format!("CHECKPOINT_HISTORY_INVALID: {e}"))?;
        if value.version != 2 { return Err("CHECKPOINT_HISTORY_INVALID: V2 version".into()); }
        Ok(HistoryPayload::V2(value))
    } else {
        let value = bincode::DefaultOptions::new().with_fixint_encoding().with_limit(MAX_BYTES)
            .reject_trailing_bytes().deserialize(payload).map_err(|e| format!("CHECKPOINT_HISTORY_INVALID: {e}"))?;
        Ok(HistoryPayload::V1(value))
    }
}
pub(crate) fn validate_mass_authority(original:&crate::data::DataSet,loaded:&crate::data::DataSet)->Result<(),String>{
    let a=original.prop.vertex.as_slice();let b=loaded.prop.vertex.as_slice();
    if a.len()!=b.len()||a.iter().zip(b).any(|(a,b)|a.mass.to_bits()!=b.mass.to_bits()||a.pdrd_body_index!=b.pdrd_body_index){return Err("CHECKPOINT_MASS_AUTHORITY_MISMATCH: original vertex mass/membership differs".into());}
    let a=original.prop.pdrd_body.as_slice();let b=loaded.prop.pdrd_body.as_slice();
    if a.len()!=b.len()||a.iter().zip(b).any(|(a,b)|a.mass_per_vertex.to_bits()!=b.mass_per_vertex.to_bits()||a.volume.to_bits()!=b.volume.to_bits()||a.rest_gram_inv!=b.rest_gram_inv||a.rest_centroid!=b.rest_centroid||a.vertex_start!=b.vertex_start||a.vertex_count!=b.vertex_count){return Err("CHECKPOINT_MASS_AUTHORITY_MISMATCH: original rigid rest/inertial metric differs".into());}
    Ok(())
}
pub(crate) fn gzip<T:serde::Serialize>(value:&T)->Result<Vec<u8>,String>{
    use std::io::Write;
    let mut encoder=flate2::write::GzEncoder::new(Vec::new(),flate2::Compression::default());
    let raw=bincode::serialize(value).map_err(|e|format!("CHECKPOINT_SERIALIZE_INVALID: {e}"))?;
    encoder.write_all(&raw).map_err(|e|e.to_string())?;encoder.finish().map_err(|e|e.to_string())
}
pub(crate) fn read_versioned(output:&Path,frame:i32,required:bool)->Result<Option<HistoryPayload>,String>{
    let path=output.join(crate::checkpoint_commit::history_name(frame));
    let meta=match std::fs::symlink_metadata(&path){Ok(m)=>m,Err(e) if e.kind()==std::io::ErrorKind::NotFound&&!required=>return Ok(None),Err(e)=>return Err(format!("CHECKPOINT_HISTORY_REQUIRED: {}: {e}",path.display()))};
    if !meta.is_file()||meta.file_type().is_symlink()||meta.len()>MAX_BYTES{return Err("CHECKPOINT_HISTORY_INVALID: unsafe/oversized sidecar".into());}
    let bytes=std::fs::read(&path).map_err(|e|format!("CHECKPOINT_HISTORY_INVALID: {e}"))?;
    let (boundary_time,payload)=crate::checkpoint_commit::decode_history(frame,&bytes)?;
    let decoded=decode_payload(payload)?;
    let h=decoded.legacy();
    if h.accepted_time.to_bits()!=boundary_time.to_bits(){return Err("CHECKPOINT_HISTORY_INVALID: frame/payload clock binding".into());}
    crate::checkpoint_values::validate_history_header(&h.kind,h.version,h.frame,frame,h.accepted_time,h.previous_dt)?;
    if !h.adaptive_start_time.is_finite()||h.adaptive_start_time<0.||h.adaptive_start_time>h.accepted_time{return Err("CHECKPOINT_ADAPTIVE_INVALID: sample clock".into());}
    Ok(Some(decoded))
}
/// Legacy callers must never silently discard V2 history.
pub(crate) fn read(output:&Path,frame:i32,required:bool)->Result<Option<History>,String>{
    match read_versioned(output,frame,required)? {
        None=>Ok(None), Some(HistoryPayload::V1(h))=>Ok(Some(h)),
        Some(HistoryPayload::V2(_))=>Err("CHECKPOINT_HISTORY_INVALID: V2 requires continuous history consumer".into()),
        Some(HistoryPayload::V3(_))=>Err("CHECKPOINT_HISTORY_INVALID: V3 requires material spin consumer".into()),
    }
}
impl HistoryPayload {
    pub(crate) fn semantic_eq(&self, other:&Self)->bool {
        match (self,other) {
            (Self::V1(a),Self::V1(b)) => a.semantic_eq(b),
            (Self::V2(a),Self::V2(b)) => a.legacy.semantic_eq(&b.legacy) && a.version==b.version && a.twist==b.twist && a.mode==b.mode,
            (Self::V3(a),Self::V3(b)) => a.legacy.semantic_eq(&b.legacy) && a.version==b.version && a.model==b.model && a.rigid_material==b.rigid_material && a.fem_material==b.fem_material,
            _ => false,
        }
    }
    pub(crate) fn legacy(&self)->&History { match self { Self::V1(h)=>h, Self::V2(h)=>&h.legacy, Self::V3(h)=>&h.legacy } }
    pub(crate) fn encode(&self)->Result<Vec<u8>,String> {
        let h=self.legacy();
        let payload=match self { Self::V1(h)=>bincode::serialize(h).map_err(|e|format!("CHECKPOINT_HISTORY_INVALID: {e}"))?, Self::V2(h)=>h.encode_payload()?, Self::V3(h)=>h.encode_payload()? };
        crate::checkpoint_commit::encode_history(h.frame,h.accepted_time,&payload)
    }
}
impl HistoryPayload {
    /// Construct a continuous sidecar from committed state and sampled actual pose.
    pub(crate) fn continuous(legacy: History, twist: crate::continuous_twist::AcceptedTwist,
        mode: TwistModeMetadata, scene: &str, joints: &[String], principal: &[f64], sealed_models: &[String], expected_selected: &[String]) -> Result<Self,String> {
        let v = HistoryV2 { version: 2, legacy, twist, mode };
        v.validate_twist(scene,joints,principal,sealed_models,expected_selected)?;
        Ok(Self::V2(v))
    }
    /// V1 has no historical winding. An explicit upgrade seeds the restored pose
    /// and records that earlier turns are unknown. V2 must never be reseeded.
    pub(crate) fn restore_twist(&self, scene: &str, joints: &[String], principal: &[f64],
        explicit_migration: Option<TwistModeMetadata>, sealed_models: &[String], expected_selected: &[String]) -> Result<Option<(crate::continuous_twist::AcceptedTwist,TwistModeMetadata)>,String> {
        match self {
            Self::V3(v) => {
                if explicit_migration.is_some(){return Err("CHECKPOINT_TWIST_INVALID: cannot migrate V3".into());}
                let t=v.twist.as_ref().ok_or("CHECKPOINT_TWIST_INVALID: V3 twist history missing")?;
                if t.legacy!=v.legacy{return Err("CHECKPOINT_TWIST_INVALID: V3 legacy boundary differs".into());}
                t.validate_twist(scene,joints,principal,sealed_models,expected_selected)?;
                Ok(Some((t.twist.clone(),t.mode.clone())))
            },
            Self::V2(v) => {
                if explicit_migration.is_some() { return Err("CHECKPOINT_TWIST_INVALID: cannot migrate V2".into()); }
                v.validate_twist(scene,joints,principal,sealed_models,expected_selected)?;
                Ok(Some((v.twist.clone(),v.mode.clone())))
            },
            Self::V1(legacy) => match explicit_migration {
                None => Ok(None),
                Some(mut mode) => {
                    mode.migration_time = Some(legacy.accepted_time);
                    let twist = crate::continuous_twist::AcceptedTwist { scene_uuid:scene.to_owned(),
                        joint_uuids:joints.to_vec(), time:legacy.accepted_time, theta:principal.to_vec(), prior_turns_known:false };
                    let v=HistoryV2 {version:2,legacy:legacy.clone(),twist,mode};
                    v.validate_twist(scene,joints,principal,sealed_models,expected_selected)?;
                    Ok(Some((v.twist,v.mode)))
                }
            }
        }
    }
}
impl HistoryV2 {
    /// Call only with phases sampled from the actual restored pose and rest frames.
    pub(crate) fn validate_twist(&self,scene:&str,joints:&[String],principal:&[f64],sealed_models:&[String],expected_selected:&[String])->Result<(),String>{
        self.twist.validate(scene,joints,self.legacy.accepted_time).map_err(str::to_owned)?;
        crate::continuous_twist::project_device(&self.twist.theta).map_err(str::to_owned)?;
        if self.legacy.scene_uuid!=scene || self.mode.effective_model!="parent_local_swing_twist_x_continuous_v2" ||
            self.mode.fit_scheme!="selected_covariance_polar_svd_v1" || self.mode.source_models.len()!=joints.len() || self.mode.source_models != sealed_models || self.mode.selected_joint_uuids != expected_selected || principal.len()!=joints.len() ||
            self.mode.migration_time.is_some_and(|t|!t.is_finite()||t<0.||t>self.legacy.accepted_time) ||
            (self.twist.prior_turns_known == self.mode.migration_time.is_some()) {
            return Err("CHECKPOINT_TWIST_INVALID: mode, identity or clock".into());
        }
        let mut selected=self.mode.selected_joint_uuids.iter();
        let mut next=selected.next();
        for (joint, model) in joints.iter().zip(sealed_models) { if next==Some(joint) { if model != "parent_local_swing_twist_x_v1" && model != "parent_local_swing_twist_x_continuous_v2" { return Err("CHECKPOINT_TWIST_INVALID: selected joint model".into()); } next=selected.next(); } }
        if next.is_some() || self.mode.selected_joint_uuids.is_empty() {
            return Err("CHECKPOINT_TWIST_INVALID: selected joint order".into());
        }
        for (&theta,&phase) in self.twist.theta.iter().zip(principal) {
            if !phase.is_finite() || phase.abs()>std::f64::consts::PI+1e-6 ||
                ((theta-phase+std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)-std::f64::consts::PI).abs()>1e-4 {
                return Err("CHECKPOINT_TWIST_INVALID: restored pose phase mismatch".into());
            }
        }
        Ok(())
    }
}
impl History {
    pub(crate) fn encode(&self)->Result<Vec<u8>,String>{let payload=bincode::serialize(self).map_err(|e|format!("CHECKPOINT_HISTORY_INVALID: {e}"))?;crate::checkpoint_commit::encode_history(self.frame,self.accepted_time,&payload)}
    /// Identity values are already authenticated by the product's input seal;
    /// this binds history to that same generation without another hash pass.
    pub(crate) fn validate_static(&self,program:&Path,sim:&crate::args::SimArgs,state:&crate::backend::State,
        owned:Option<&crate::joint::OwnedJointScene>,soft:Option<&crate::soft_joint::OwnedSoftScene>,
        curves:Option<&crate::restoring_native::Installed>,bodies:usize,authored_dt:f32)->Result<(),String>{
        if self.frame!=state.curr_frame||self.accepted_time.to_bits()!=state.time.to_bits()||self.previous_dt.to_bits()!=state.prev_dt.to_bits()||
            self.sealed_frames!=sim.frames||self.fps.to_bits()!=sim.fps.to_bits()||
            self.param_bytes!=std::fs::read(program.join("param.toml")).map_err(|e|format!("CHECKPOINT_INPUT_INVALID: {e}"))? {
            return Err("CHECKPOINT_HISTORY_INVALID: State/parameter boundary mismatch".into());
        }
        if let Some(o)=owned {if self.scene_uuid!=o.scene_uuid||self.scene_inputs_sha256!=o.scene_inputs_sha256{return Err("CHECKPOINT_GENERATION_MISMATCH: sealed input differs".into());}}
        if !self.adaptive_start_time.is_finite()||self.adaptive_start_time<0.||self.adaptive_start_time>self.accepted_time||self.adaptive_enabled!=crate::backend::adaptive_cg_retry_enabled(){return Err("CHECKPOINT_ADAPTIVE_INVALID: policy/sample clock differs".into());}
        self.adaptive.validate_checkpoint(authored_dt)?;
        let root=owned.filter(|o|!o.roots.is_empty()).map(|o|{
            let source=self.runtime.root.as_ref().map_or(0,|r|r.source_generation);
            crate::joint_root_motion_state::RootState::new(if source==1{2}else{1},o.roots.clone()).map_err(|e|e.to_string())
        }).transpose()?;
        let fem=soft.and_then(|s|s.control_events.as_ref()).map(|p|{
            let source=self.runtime.fem.as_ref().map(|s|s.source_launch_id.as_str());
            let launch=if source==Some("checkpoint-preflight"){"checkpoint-preflight-2"}else{"checkpoint-preflight"};
            crate::fem_control_events::Runtime::new(p.bodies.clone(),&p.positions,launch.into())
        }).transpose()?;
        crate::driver::checkpoint::validate(&self.runtime,state.time,bodies,root.as_ref(),fem.as_ref(),curves.map_or(&[],|c|c.bindings.as_slice()))
    }
    pub(crate) fn semantic_eq(&self,old:&Self)->bool{
        let mut a=self.clone();let mut b=old.clone();
        // Attempt identities belong to their processes, not accepted physics.
        if let Some(r)=&mut a.runtime.root{r.source_generation=1;}if let Some(r)=&mut b.runtime.root{r.source_generation=1;}
        if let Some(r)=&mut a.runtime.fem{r.source_launch_id.clear();}if let Some(r)=&mut b.runtime.fem{r.source_launch_id.clear();}
        a==b
    }
}

#[cfg(test)] mod continuous_codec_tests {
    use super::*;
    fn history() -> History {
        History { kind:"ppf.bone_checkpoint_history".into(), version:1, frame:3, accepted_time:1.0, previous_dt:0.1,
            sealed_frames:9, fps:24.0, param_bytes:vec![1,2], scene_uuid:"scene".into(), scene_inputs_sha256:"sealed".into(),
            filtered_links:0, joint_accepted:4, adaptive:crate::driver::adaptive_controller::Controller::new(0.1),
            adaptive_enabled:false, adaptive_start_time:0.0,
            runtime:crate::driver::checkpoint::Snapshot {running_rotation:vec![],rotation_seeded:false,root:None,fem:None,fem_steps:0,restoring:None,restoring_commit_count:0,restoring_commit_time:None} }
    }
    fn v2()->HistoryV2 { HistoryV2 {version:2,legacy:history(),twist:crate::continuous_twist::AcceptedTwist {
        scene_uuid:"scene".into(),joint_uuids:vec!["joint".into()],time:1.0,theta:vec![std::f64::consts::TAU+0.25],prior_turns_known:true},
        mode:TwistModeMetadata {fit_scheme:"selected_covariance_polar_svd_v1".into(),effective_model:"parent_local_swing_twist_x_continuous_v2".into(),source_models:vec!["parent_local_swing_twist_x_v1".into()],selected_joint_uuids:vec!["joint".into()],migration_time:None}} }
    #[test] fn v3_roundtrip_preserves_bend_and_refuses_twist_consumer() {
        let v=HistoryV3 {version:3,legacy:history(),model:crate::material_spin_history::MODEL.into(),
            rigid_material:Some(crate::material_spin_history::AcceptedMaterialSpin {scene_uuid:"scene".into(),joint_uuids:vec!["joint".into()],time:1.0,generation:4,
                rows:vec![crate::material_spin_history::MaterialSpinRow {q_wxyz:[1.,0.,0.,0.],a:[12.566370614359172,3.490658503988659,0.]}]}),fem_material:None,twist:None};
        let bytes=v.encode_payload().unwrap();
        let decoded=decode_payload(&bytes).unwrap();
        assert_eq!(decoded,HistoryPayload::V3(v.clone()));
        assert!(v.validate_material("scene", &["joint".into()]).is_ok());
        assert!(v.validate_material("scene", &["other".into()]).is_err());
        assert!(decoded.restore_twist("scene",&["joint".into()],&[0.],None,&[],&[]).is_err());
        for n in [8,9,bytes.len()-1] {assert!(decode_payload(&bytes[..n]).is_err());}
    }
    #[test] fn positional_v1_and_v2_roundtrip_and_corruption() {
        let old=history(); let bytes=bincode::serialize(&old).unwrap();
        assert_eq!(decode_payload(&bytes).unwrap(),HistoryPayload::V1(old));
        let new=v2();let bytes=new.encode_payload().unwrap();
        assert_eq!(decode_payload(&bytes).unwrap(),HistoryPayload::V2(new.clone()));
        for len in [8,9,bytes.len()-1] {assert!(decode_payload(&bytes[..len]).is_err());}
        let mut wrong=bytes.clone();wrong[8]=3;assert!(decode_payload(&wrong).is_err());
        let mut trailing=bytes.clone();trailing.push(0);assert!(decode_payload(&trailing).is_err());
        assert!(new.validate_twist("scene", &["joint".into()], &[0.25], &["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_ok());
        assert!(new.validate_twist("scene", &["joint".into()], &[0.5], &["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
        assert!(new.validate_twist("other", &["joint".into()], &[0.25], &["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
    }
    #[test] fn unpublished_legacy_v2_without_fit_scheme_is_rejected() {
        #[derive(serde::Serialize)] struct OldMode {effective_model:String,source_models:Vec<String>,selected_joint_uuids:Vec<String>,migration_time:Option<f64>}
        #[derive(serde::Serialize)] struct OldV2 {version:u32,legacy:History,twist:crate::continuous_twist::AcceptedTwist,mode:OldMode}
        let current=v2();
        let old=OldV2 {version:2,legacy:current.legacy,twist:current.twist,mode:OldMode {effective_model:current.mode.effective_model,source_models:current.mode.source_models,selected_joint_uuids:current.mode.selected_joint_uuids,migration_time:current.mode.migration_time}};
        let mut bytes=V2_MAGIC.to_vec(); bytes.extend(bincode::serialize(&old).unwrap());
        assert!(decode_payload(&bytes).is_err());
    }
    #[test] fn restore_requires_explicit_migration_and_checks_pose_order_clock() {
        let old=HistoryPayload::V1(history()); let joints=vec!["joint".to_string()];
        assert!(old.restore_twist("scene",&joints,&[0.25],None,&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).unwrap().is_none());
        let mode=v2().mode;
        let (migrated, metadata)=old.restore_twist("scene",&joints,&[0.25],Some(mode.clone()),&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).unwrap().unwrap();
        assert_eq!(migrated.theta,vec![0.25]); assert!(!migrated.prior_turns_known);
        assert_eq!(metadata.migration_time,Some(1.0));
        let fresh=HistoryPayload::continuous(history(),migrated,metadata,"scene",&joints,&[0.25],&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).unwrap();
        assert!(fresh.restore_twist("scene",&joints,&[0.25],None,&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_ok());
        assert!(fresh.restore_twist("scene",&joints,&[0.25],Some(mode),&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
        assert!(fresh.restore_twist("scene",&["other".into()],&[0.25],None,&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
        assert!(fresh.restore_twist("scene",&joints,&[f64::NAN],None,&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
        assert!(fresh.restore_twist("scene",&joints,&[0.5],None,&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
        assert!(fresh.restore_twist("scene",&joints,&[0.25],None,&["incorrect".into()], &["joint".into()]).is_err());
        let mut bad=v2(); bad.twist.theta[0]=1024.25;
        assert!(HistoryPayload::V2(bad).restore_twist("scene",&joints,&[0.25],None,&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
        assert!(fresh.restore_twist("scene",&joints,&[0.25],None,&["parent_local_swing_twist_x_v1".into()],&[]).is_err());
        assert!(fresh.restore_twist("scene",&joints,&[0.25],None,&["parent_local_swing_twist_x_v1".into()],&["joint".into(),"added".into()]).is_err());
        let mut bad=v2(); bad.twist.prior_turns_known=false;
        assert!(HistoryPayload::V2(bad).restore_twist("scene",&joints,&[0.25],None,&["parent_local_swing_twist_x_v1".into()],&["joint".into()]).is_err());
        let mut bad=v2(); bad.mode.fit_scheme="old_iterative_fit".into();
        assert!(HistoryPayload::V2(bad).restore_twist("scene",&joints,&[0.25],None,&["parent_local_swing_twist_x_v1".into()],&["joint".into()]).is_err());
        let mut bad=v2(); bad.twist.time=2.0;
        assert!(HistoryPayload::V2(bad).restore_twist("scene",&joints,&[0.25],None,&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
        let mut bad=v2(); bad.mode.selected_joint_uuids=vec!["wrong".into()];
        assert!(HistoryPayload::V2(bad).restore_twist("scene",&joints,&[0.25],None,&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
        assert!(old.restore_twist("other",&joints,&[0.25],Some(v2().mode),&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
        assert!(old.restore_twist("scene",&joints,&[f64::INFINITY],Some(v2().mode),&["parent_local_swing_twist_x_v1".into()], &["joint".into()]).is_err());
    }
    #[test] #[ignore = "requires an actual private checkpoint"]
    fn inspect_actual_checkpoint() {
        let path=std::env::var_os("PPF_PRIVATE_CHECKPOINT_DIR").expect("private output directory");
        let frame:i32=std::env::var("PPF_PRIVATE_CHECKPOINT_FRAME").expect("frame").parse().unwrap();
        let h=read_versioned(std::path::Path::new(&path),frame,true).unwrap().unwrap();
        match h {
            HistoryPayload::V1(v) => println!("CHECKPOINT_V1 frame={} time={} previous_dt={} sealed_frames={} fps={} scene={} filtered_links={} accepted={} adaptive={} restoring_present={} root_present={} fem_present={} running_rotation={:?}",v.frame,v.accepted_time,v.previous_dt,v.sealed_frames,v.fps,v.scene_uuid,v.filtered_links,v.joint_accepted,v.adaptive_enabled,v.runtime.restoring.is_some(),v.runtime.root.is_some(),v.runtime.fem.is_some(),v.runtime.running_rotation),
            HistoryPayload::V2(v) => println!("CHECKPOINT_ACTUAL frame={} time={} prior_known={} migration_time={:?} joints={:?} theta={:?} running_rotation={:?}",frame,v.legacy.accepted_time,v.twist.prior_turns_known,v.mode.migration_time,v.twist.joint_uuids,v.twist.theta,v.legacy.runtime.running_rotation),
            HistoryPayload::V3(v) => println!("CHECKPOINT_MATERIAL_SPIN frame={} time={} rigid={:?} fem={:?}",frame,v.legacy.accepted_time,v.rigid_material,v.fem_material),
        }
    }
    #[test] #[ignore = "mutates an explicitly supplied private test checkpoint"]
    fn mutate_private_checkpoint_phase() {
        let path=std::path::PathBuf::from(std::env::var_os("PPF_PRIVATE_CHECKPOINT_DIR").expect("private output directory"));
        let frame:i32=std::env::var("PPF_PRIVATE_CHECKPOINT_FRAME").expect("frame").parse().unwrap();
        let h=read_versioned(&path,frame,true).unwrap().unwrap();
        let HistoryPayload::V2(mut v)=h else {panic!("expected V2")};
        v.twist.theta[0]+=0.5;
        std::fs::write(path.join(crate::checkpoint_commit::history_name(frame)),HistoryPayload::V2(v).encode().unwrap()).unwrap();
    }
}
