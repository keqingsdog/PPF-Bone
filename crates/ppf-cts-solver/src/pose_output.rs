//! Versioned measurements of already published accepted output vertex frames.
//! Never feeds a pose back to the solver and never claims constraint authority.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, io::Write, path::Path};

pub type Result<T> = std::result::Result<T, String>;
pub const KIND: &str = "PdrdFramePoseMeasurement";
pub fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn digest(s: &str) -> bool { s.len()==64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) }
fn uuid(s: &str) -> bool {
    s.len()==36 && s.bytes().enumerate().all(|(i,c)| if [8,13,18,23].contains(&i) { c==b'-' } else { c.is_ascii_digit() || (b'a'..=b'f').contains(&c) })
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Binding { pub body_uuid: String, pub runtime_body_index: u32 }
/// Native PdrdRigidState fields, copied after fit of the exact vertex frame.
#[derive(Debug, Clone)]
pub struct NativeMeasurement { pub anchor: [f32;3], pub x: [f32;3], pub rotation: [f32;9], pub count: u32 }
/// Internal read-only actual-geometry fit; centroid remains f64 through export.
#[derive(Debug,Clone)]
pub struct GeometryMeasurement { pub center:[f64;3], pub rotation:[f32;9], pub count:u32 }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Source { InitialFrame, AcceptedEndpoint, InterpolatedAcceptedBracket }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BodyPose {
    pub body_uuid: String, pub runtime_body_index: u32, pub vertex_count: u32,
    pub centroid_m: [f64;3], pub rotation_column_major: [f32;9],
}
impl BodyPose {
    /// World solver-y-up meters from a REST-CENTERED solver-basis vector.
    /// This is not the original Blender proxy object's local coordinate.
    pub fn transform_rest_point(&self, rest:[f64;3]) -> Result<[f64;3]> {
        rotation(&self.rotation_column_major)?;
        if !rest.iter().chain(&self.centroid_m).all(|v|v.is_finite()) {return Err("nonfinite frame point".into());}
        let world=std::array::from_fn(|k| self.centroid_m[k]+(0..3).map(|j|f64::from(self.rotation_column_major[3*j+k])*rest[j]).sum::<f64>());
        if !world.iter().all(|v|v.is_finite()) {return Err("frame transform overflow".into());} Ok(world)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PoseFrame {
    pub version: u32, pub kind: String, pub scene_uuid: String,
    pub scene_inputs_sha256: String, pub vertex_frame_sha256: String,
    pub vertex_frame_byte_length: u64, pub frame_id: u32,
    pub frame_time_seconds: f64, pub accepted_bracket_start_seconds: f64,
    pub accepted_solver_time_seconds: f64, pub accepted_steps: u64,
    pub source: Source,
    /// Fixed semantics, validated on encode AND decode; not a solver state.
    pub measurement: String, pub coordinate_system: String,
    pub constrained_root_state_authority: String,
    pub bodies: Vec<BodyPose>,
}
fn rotation(r: &[f32;9]) -> Result<()> {
    if !r.iter().all(|x| x.is_finite()) { return Err("nonfinite rotation".into()); }
    let dot = |a:usize,b:usize| (0..3).map(|k| f64::from(r[3*a+k])*f64::from(r[3*b+k])).sum::<f64>();
    for a in 0..3 { for b in 0..3 { if (dot(a,b)-if a==b {1.0} else {0.0}).abs()>5e-4 { return Err("nonorthonormal rotation".into()); } } }
    let d=f64::from(r[0])*(f64::from(r[4])*f64::from(r[8])-f64::from(r[7])*f64::from(r[5]))
        -f64::from(r[3])*(f64::from(r[1])*f64::from(r[8])-f64::from(r[7])*f64::from(r[2]))
        +f64::from(r[6])*(f64::from(r[1])*f64::from(r[5])-f64::from(r[4])*f64::from(r[2]));
    if (d-1.0).abs()>5e-4 { return Err("improper rotation".into()); } Ok(())
}
impl PoseFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn from_measurements(scene_uuid:String, scene_inputs_sha256:String, vertices:&[u8],
        frame_id:u32, time:f64, start:f64, end:f64, steps:u64, bindings:&[Binding],
        native:&[NativeMeasurement], world_scaling:f32) -> Result<Self> {
        if !world_scaling.is_finite() || world_scaling<=0.0 { return Err("invalid world scaling".into()); }
        let geometry:Vec<_>=native.iter().map(|n|GeometryMeasurement {
            center:std::array::from_fn(|k|f64::from(n.anchor[k])+f64::from(n.x[k])),rotation:n.rotation,count:n.count }).collect();
        Self::from_geometry_measurements(scene_uuid,scene_inputs_sha256,vertices,frame_id,time,start,end,steps,bindings,&geometry,world_scaling)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn from_geometry_measurements(scene_uuid:String, scene_inputs_sha256:String, vertices:&[u8],
        frame_id:u32,time:f64,start:f64,end:f64,steps:u64,bindings:&[Binding],
        native:&[GeometryMeasurement],world_scaling:f32)->Result<Self> {
        if !world_scaling.is_finite()||world_scaling<=0.0 {return Err("invalid world scaling".into());}
        if bindings.len()!=native.len() { return Err("body count mismatch".into()); }
        let mut sorted=bindings.to_vec(); sorted.sort_by_key(|b| b.runtime_body_index);
        let mut bodies=Vec::with_capacity(sorted.len());
        for (i,b) in sorted.iter().enumerate() {
            if b.runtime_body_index as usize!=i { return Err("body indices must be dense and unique".into()); }
            let n=&native[i];
            if !n.center.iter().all(|x|x.is_finite()) { return Err("nonfinite native centroid".into()); }
            bodies.push(BodyPose {body_uuid:b.body_uuid.clone(), runtime_body_index:i as u32,
                vertex_count:n.count, centroid_m:std::array::from_fn(|k|n.center[k]/f64::from(world_scaling)), rotation_column_major:n.rotation});
        }
        let source=if frame_id==0 && steps==0 {Source::InitialFrame} else if time==end {Source::AcceptedEndpoint} else {Source::InterpolatedAcceptedBracket};
        let value=Self {version:1,kind:KIND.into(),scene_uuid,scene_inputs_sha256,vertex_frame_sha256:hash(vertices),vertex_frame_byte_length:vertices.len() as u64,
            frame_id,frame_time_seconds:time,accepted_bracket_start_seconds:start,accepted_solver_time_seconds:end,accepted_steps:steps,source,
            measurement:"read_only_native_pdrd_fit_of_vertex_frame".into(),coordinate_system:"solver_y_up_meters".into(),constrained_root_state_authority:"not_reported".into(),bodies};
        value.verify(vertices)?; Ok(value)
    }
    pub fn verify(&self, vertices:&[u8]) -> Result<()> {
        if self.version!=1 || self.kind!=KIND || self.measurement!="read_only_native_pdrd_fit_of_vertex_frame" || self.coordinate_system!="solver_y_up_meters" || self.constrained_root_state_authority!="not_reported" { return Err("unsupported pose contract".into()); }
        if !uuid(&self.scene_uuid) || !digest(&self.scene_inputs_sha256) || !digest(&self.vertex_frame_sha256) { return Err("invalid identity".into()); }
        if vertices.is_empty() || vertices.len()%12!=0 || self.vertex_frame_byte_length!=vertices.len() as u64 || self.vertex_frame_sha256!=hash(vertices) { return Err("vertex frame hash/length mismatch".into()); }
        for chunk in vertices.chunks_exact(4) { if !f32::from_le_bytes(chunk.try_into().unwrap()).is_finite() {return Err("nonfinite vertex frame".into());} }
        let (t,a,b)=(self.frame_time_seconds,self.accepted_bracket_start_seconds,self.accepted_solver_time_seconds);
        if ![t,a,b].iter().all(|x| x.is_finite()) || a<0.0 || t<a || t>b {return Err("invalid accepted time bracket".into());}
        match self.source {
            Source::InitialFrame if self.frame_id==0 && self.accepted_steps==0 && t==0.0 && a==0.0 && b==0.0 => {},
            Source::AcceptedEndpoint if self.accepted_steps>0 && self.frame_id>0 && t==b && b>a => {},
            Source::InterpolatedAcceptedBracket if self.accepted_steps>0 && self.frame_id>0 && t>=a && t<b && b>a => {},
            _ => return Err("source/acceptance metadata mismatch".into()),
        }
        if self.bodies.is_empty() {return Err("empty rigid identity map".into());}
        let mut seen=HashSet::new(); let mut count=0u64;
        for (i,p) in self.bodies.iter().enumerate() {
            if !uuid(&p.body_uuid) || !seen.insert(&p.body_uuid) || p.runtime_body_index as usize!=i {return Err("invalid/duplicate rigid identity".into());}
            if p.vertex_count<4 || !p.centroid_m.iter().all(|x| x.is_finite()) {return Err("invalid rigid geometry".into());}
            rotation(&p.rotation_column_major)?; count+=u64::from(p.vertex_count);
        }
        if count>vertices.len() as u64/12 {return Err("rigid vertex count exceeds frame".into());} Ok(())
    }
}
pub fn encode(frame:&PoseFrame, vertices:&[u8]) -> Result<Vec<u8>> { frame.verify(vertices)?; let mut out=Vec::new(); ciborium::into_writer(frame,&mut out).map_err(|e| e.to_string())?; Ok(out) }
/// Expected frame/scene identity supplied by the consumer, never inferred from the sidecar.
pub fn decode(bytes:&[u8], vertices:&[u8], frame_id:u32, scene_uuid:&str, scene_hash:&str, expected:&[Binding]) -> Result<PoseFrame> {
    let mut cursor=std::io::Cursor::new(bytes); let f:PoseFrame=ciborium::from_reader(&mut cursor).map_err(|e| e.to_string())?;
    if cursor.position()!=bytes.len() as u64 {return Err("trailing pose data".into());}
    f.verify(vertices)?;
    if f.frame_id!=frame_id || f.scene_uuid!=scene_uuid || f.scene_inputs_sha256!=scene_hash {return Err("frame/scene identity mismatch".into());}
    let mut expected=expected.to_vec(); expected.sort_by_key(|b| b.runtime_body_index);
    if expected.len()!=f.bodies.len() || expected.iter().zip(&f.bodies).any(|(b,p)| b.body_uuid!=p.body_uuid || b.runtime_body_index!=p.runtime_body_index) {return Err("body identity/count differs from sealed binding".into());} Ok(f)
}
/// Commit marker follows the already published vertex frame. No overwriting a
/// different generation; duplicate publication of identical bytes is idempotent.
pub fn publish(output:&Path, frame:&PoseFrame, vertices:&[u8]) -> Result<()> {
    if std::fs::read(output.join(format!("vert_{}.bin",frame.frame_id))).map_err(|e|e.to_string())?!=vertices {return Err("published vertex generation differs".into());}
    let bytes=encode(frame,vertices)?; let path=output.join(format!("rigid_pose_{}.cbor",frame.frame_id));
    if path.exists() {return if std::fs::read(&path).map_err(|e| e.to_string())?==bytes {Ok(())} else {Err("conflicting pose generation".into())};}
    let tmp=output.join(format!("rigid_pose_{}.cbor.tmp",frame.frame_id));
    let mut file=std::fs::OpenOptions::new().create_new(true).write(true).open(&tmp).map_err(|e|e.to_string())?;
    let written=(|| {file.write_all(&bytes)?;file.sync_all()})().map_err(|e| e.to_string()); drop(file);
    if let Err(e)=written {let _=std::fs::remove_file(&tmp); return Err(e);}
    if let Err(e)=std::fs::rename(&tmp,&path) {let _=std::fs::remove_file(&tmp); return Err(e.to_string());} Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn uid(n:u32)->String {format!("{n:08x}-0000-0000-0000-000000000000")}
    fn bindings()->Vec<Binding> {vec![Binding{body_uuid:uid(2),runtime_body_index:0},Binding{body_uuid:uid(3),runtime_body_index:1}]}
    fn vertices()->Vec<u8> {(0..24).flat_map(|i|(i as f32/100.0).to_le_bytes()).collect()}
    fn native()->Vec<NativeMeasurement> {vec![NativeMeasurement{anchor:[1000.0,0.0,0.0],x:[0.125,0.0,0.0],rotation:[1.,0.,0.,0.,1.,0.,0.,0.,1.],count:4};2]}
    fn frame()->PoseFrame {PoseFrame::from_measurements(uid(1),"a".repeat(64),&vertices(),7,0.25,0.24,0.26,19,&bindings(),&native(),1.).unwrap()}
    fn decoded(bytes:&[u8],vertices:&[u8])->Result<PoseFrame> {decode(bytes,vertices,7,&uid(1),&"a".repeat(64),&bindings())}
    #[test] fn roundtrip_reorders_bindings_and_preserves_anchor_precision() {
        let mut b=bindings();b.reverse();
        let f=PoseFrame::from_measurements(uid(1),"a".repeat(64),&vertices(),7,0.25,0.24,0.26,19,&b,&native(),1.).unwrap();
        assert_eq!(f.bodies[0].centroid_m[0],1000.125);
        assert_eq!(f.source,Source::InterpolatedAcceptedBracket);
        assert_eq!(decoded(&encode(&f,&vertices()).unwrap(),&vertices()).unwrap(),f);
    }
    #[test] fn rotation_columns_map_rest_centered_basis_without_proxy_rotation_guessing() {
        let mut f=frame();f.bodies[0].rotation_column_major=[0.,1.,0.,-1.,0.,0.,0.,0.,1.];
        f.verify(&vertices()).unwrap();
        assert_eq!(f.bodies[0].transform_rest_point([1.,0.,0.]).unwrap(),[1000.125,1.,0.]);
        assert_eq!(f.bodies[0].transform_rest_point([0.,1.,0.]).unwrap(),[999.125,0.,0.]);
        assert!(f.bodies[0].transform_rest_point([f64::NAN,0.,0.]).is_err());
    }
    #[test] fn identity_frame_hash_time_and_count_refuse_mismatches() {
        let v=vertices(); let f=frame();let bytes=encode(&f,&v).unwrap();
        let mut changed=v.clone();changed[1]^=1;assert!(decoded(&bytes,&changed).is_err());
        assert!(decode(&bytes,&v,8,&uid(1),&"a".repeat(64),&bindings()).is_err());
        assert!(decode(&bytes,&v,7,&uid(9),&"a".repeat(64),&bindings()).is_err());
        assert!(decode(&bytes,&v,7,&uid(1),&"b".repeat(64),&bindings()).is_err());
        let mut changed_bindings=bindings();changed_bindings[0].body_uuid=uid(8);
        assert!(decode(&bytes,&v,7,&uid(1),&"a".repeat(64),&changed_bindings).is_err());
        assert!(decode(&bytes,&v,7,&uid(1),&"a".repeat(64),&bindings()[..1]).is_err());
        assert!(PoseFrame::from_measurements(uid(1),"a".repeat(64),&v,7,0.25,0.24,0.26,19,&bindings(),&native()[..1],1.).is_err());
        for bad in [f64::NAN,0.23,0.27] {let mut p=f.clone();p.frame_time_seconds=bad;assert!(p.verify(&v).is_err());}
        let mut p=f.clone();p.accepted_steps=0;assert!(p.verify(&v).is_err());
    }
    #[test] fn improper_nonfinite_shear_and_bad_uuids_refuse_without_projection() {
        let v=vertices();let good=frame();
        for value in [f32::NAN,f32::INFINITY,-1.,2.] {let mut f=good.clone();f.bodies[0].rotation_column_major[0]=value;assert!(f.verify(&v).is_err());}
        let mut f=good.clone();f.bodies[0].rotation_column_major[3]=0.1;assert!(f.verify(&v).is_err());
        let mut f=good.clone();f.bodies[1].body_uuid=f.bodies[0].body_uuid.clone();assert!(f.verify(&v).is_err());
        let mut f=good.clone();f.bodies[0].body_uuid="not a uuid".into();assert!(f.verify(&v).is_err());
        let mut f=good.clone();f.bodies[1].runtime_body_index=0;assert!(f.verify(&v).is_err());
        let mut f=good.clone();f.bodies[0].vertex_count=5;assert!(f.verify(&v).is_err());
        let mut n=native();n[0].x[0]=f32::NAN;assert!(PoseFrame::from_measurements(uid(1),"a".repeat(64),&v,7,0.25,0.24,0.26,19,&bindings(),&n,1.).is_err());
    }
    #[test] fn initial_and_endpoint_sources_are_explicit() {
        let init=PoseFrame::from_measurements(uid(1),"a".repeat(64),&vertices(),0,0.,0.,0.,0,&bindings(),&native(),1.).unwrap();assert_eq!(init.source,Source::InitialFrame);
        let endpoint=PoseFrame::from_measurements(uid(1),"a".repeat(64),&vertices(),7,0.26,0.24,0.26,19,&bindings(),&native(),1.).unwrap();assert_eq!(endpoint.source,Source::AcceptedEndpoint);
        let mut f=frame();f.source=Source::AcceptedEndpoint;assert!(f.verify(&vertices()).is_err());
    }
    #[test] fn codec_refuses_unknown_versions_trailing_data_and_untrusted_matrix() {
        let v=vertices();let mut f=frame();f.version=2;let mut bytes=Vec::new();ciborium::into_writer(&f,&mut bytes).unwrap();assert!(decoded(&bytes,&v).is_err());
        let mut bytes=encode(&frame(),&v).unwrap();bytes.push(0);assert!(decoded(&bytes,&v).is_err());
        let mut f=frame();f.bodies[0].rotation_column_major[0]=-1.;let mut bytes=Vec::new();ciborium::into_writer(&f,&mut bytes).unwrap();assert!(decoded(&bytes,&v).is_err());
    }
    #[test] fn publication_requires_real_vertex_file_and_never_relabels_a_generation() {
        let dir=tempfile::tempdir().unwrap();let v=vertices();let f=frame();
        assert!(publish(dir.path(),&f,&v).is_err());
        std::fs::write(dir.path().join("vert_7.bin"),&v).unwrap();publish(dir.path(),&f,&v).unwrap();publish(dir.path(),&f,&v).unwrap();
        let bytes=std::fs::read(dir.path().join("rigid_pose_7.cbor")).unwrap();assert_eq!(decoded(&bytes,&v).unwrap(),f);
        let mut changed=f.clone();changed.bodies[0].centroid_m[0]+=1.;assert!(publish(dir.path(),&changed,&v).is_err());
        let mut v2=v.clone();v2[2]^=1;std::fs::write(dir.path().join("vert_7.bin"),&v2).unwrap();assert!(publish(dir.path(),&f,&v).is_err());
        assert!(!dir.path().join("rigid_pose_7.cbor.tmp").exists());
    }
}
