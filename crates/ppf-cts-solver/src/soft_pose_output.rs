//! Read-only material frames measured from the exact published accepted f32
//! vertex bytes (solver y-up meters). Never writes geometry or solver state.
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, io::Write, path::Path};
#[path="material_frame_rotation.rs"] mod material_frame_rotation;
pub type Result<T> = std::result::Result<T, String>;
pub const KIND: &str = "FEMMaterialFramePoseMeasurement";
pub const MEASUREMENT: &str = "read_only_fem_material_polar_of_vertex_frame";
pub fn bindings_from_owned(owned:&crate::soft_joint::OwnedSoftScene)->Result<Vec<MaterialBinding>> {
    owned.bodies.iter().filter(|b|b.kind=="FEM_TET_MATERIAL").map(|b|Ok(MaterialBinding {
        body_uuid:b.body_uuid.clone(),tet_index:b.tet_index.ok_or("missing FEM tet")?,
        vertex_indices:b.vertex_indices.ok_or("missing FEM vertex namespace")?,
        dm_inv_column_major:b.dm_inv_column_major.ok_or("missing FEM Dm inverse")?,
        center_bary:b.center_bary.ok_or("missing FEM center barycentric")?,
        frame_rest_column_major:b.frame_rest_column_major.ok_or("missing FEM rest frame")?,
    })).collect()
}
fn hash(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }
fn digest(s: &str) -> bool { s.len()==64 && s.bytes().all(|c|c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) }
fn uuid(s: &str) -> bool {s.len()==36 && s.bytes().enumerate().all(|(i,c)|if [8,13,18,23].contains(&i){c==b'-'}else{c.is_ascii_digit()||(b'a'..=b'f').contains(&c)})}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MaterialBinding {
    pub body_uuid: String, pub tet_index: u32, pub vertex_indices: [u32;4],
    pub dm_inv_column_major: [f64;9], pub center_bary: [f64;4],
    pub frame_rest_column_major: [f64;9],
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BodyPose {
    pub body_uuid:String, pub body_type:String, pub tet_index:u32,
    pub vertex_indices:[u32;4],
    /// Representative material center defined by center_bary; the field name
    /// matches the bone return transform vocabulary, not a surface centroid fit.
    pub centroid_m:[f64;3],
    pub rotation_column_major:[f64;9],
}
pub(crate) fn root_pose(p:&BodyPose)->crate::joint_root_motion_wire::RootPose {
    let r=Matrix3::from_column_slice(&p.rotation_column_major);
    let q=nalgebra::UnitQuaternion::from_matrix(&r);
    let q=q.quaternion();
    crate::joint_root_motion_wire::RootPose{center_world_m:p.centroid_m,rotation_world_wxyz:[q.w,q.i,q.j,q.k]}
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PoseFrame {
    pub version:u32, pub kind:String, pub scene_uuid:String,
    pub scene_inputs_sha256:String, pub fem_payload_sha256:String,
    pub vertex_frame_sha256:String, pub vertex_frame_byte_length:u64,
    pub frame_id:u32, pub frame_time_seconds:f64,
    pub accepted_bracket_start_seconds:f64, pub accepted_solver_time_seconds:f64,
    pub accepted_steps:u64, pub source:crate::pose_output::Source,
    pub measurement:String, pub coordinate_system:String,
    pub constrained_root_state_authority:String, pub bodies:Vec<BodyPose>,
    #[serde(default,deserialize_with="present",skip_serializing_if="Option::is_none")]
    pub control_program_sha256:Option<String>,
    #[serde(default,deserialize_with="present",skip_serializing_if="Option::is_none")]
    pub control_segments:Option<Vec<crate::fem_control_events::ControlSegment>>,
}
fn present<'de,D:serde::Deserializer<'de>,T:Deserialize<'de>>(d:D)->std::result::Result<Option<T>,D::Error>{T::deserialize(d).map(Some)}
fn rotation(r:&[f64;9])->Result<()> {
    let m=Matrix3::from_column_slice(r);
    if !m.iter().all(|v|v.is_finite()) || (m.transpose()*m-Matrix3::identity()).amax()>5e-4 || (m.determinant()-1.).abs()>5e-4 {return Err("invalid proper material rotation".into());} Ok(())
}
pub(crate) fn measure(b:&MaterialBinding, vertices:&[u8])->Result<BodyPose> {
    if !uuid(&b.body_uuid) || b.vertex_indices.iter().copied().collect::<HashSet<_>>().len()!=4 || !b.center_bary.iter().all(|v|v.is_finite()) || (b.center_bary.iter().sum::<f64>()-1.).abs()>1e-8 {return Err("invalid material binding identity/barycentrics".into());}
    rotation(&b.frame_rest_column_major)?;
    let dm=Matrix3::from_column_slice(&b.dm_inv_column_major);
    if !dm.iter().all(|v|v.is_finite()) || !dm.determinant().is_finite() || dm.determinant()==0. {return Err("singular/nonfinite reference tet".into());}
    let mut x=[Vector3::zeros();4];
    for (i,index) in b.vertex_indices.iter().enumerate() {
        let offset=(*index as usize).checked_mul(12).ok_or("vertex offset overflow")?;
        let bytes=vertices.get(offset..offset+12).ok_or("material vertex outside actual output namespace")?;
        x[i]=Vector3::from_fn(|k,_| f32::from_le_bytes(bytes[k*4..k*4+4].try_into().unwrap()) as f64);
    }
    let ds=Matrix3::from_columns(&[x[1]-x[0],x[2]-x[0],x[3]-x[0]]);
    let f=ds*dm;
    // Read exactly this accepted geometry. The SO(3) factor remains defined
    // through an orientation reversal when its rotation is unique.
    let r=material_frame_rotation::proper_rotation(&f)?*Matrix3::from_column_slice(&b.frame_rest_column_major);
    let mut rotation_column_major=[0.;9]; rotation_column_major.copy_from_slice(r.as_slice()); rotation(&rotation_column_major)?;
    let center=x.iter().zip(b.center_bary).fold(Vector3::zeros(),|acc,(p,w)|acc+p*w);
    if !center.iter().all(|v|v.is_finite()){return Err("material center overflow".into());}
    Ok(BodyPose {body_uuid:b.body_uuid.clone(),body_type:"FEM_TET_MATERIAL".into(),tet_index:b.tet_index,vertex_indices:b.vertex_indices,centroid_m:center.into(),rotation_column_major})
}
impl PoseFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn from_vertices(scene_uuid:String,scene_inputs_sha256:String,fem_payload_sha256:String,vertices:&[u8],frame_id:u32,time:f64,start:f64,end:f64,steps:u64,bindings:&[MaterialBinding])->Result<Self> {
        verify_vertices(vertices)?;
        let bodies=bindings.iter().map(|b|measure(b,vertices)).collect::<Result<Vec<_>>>()?;
        let source=if frame_id==0&&steps==0 {crate::pose_output::Source::InitialFrame} else if time==end {crate::pose_output::Source::AcceptedEndpoint} else {crate::pose_output::Source::InterpolatedAcceptedBracket};
        let f=Self {version:1,kind:KIND.into(),scene_uuid,scene_inputs_sha256,fem_payload_sha256,vertex_frame_sha256:hash(vertices),vertex_frame_byte_length:vertices.len() as u64,frame_id,frame_time_seconds:time,accepted_bracket_start_seconds:start,accepted_solver_time_seconds:end,accepted_steps:steps,source,measurement:MEASUREMENT.into(),coordinate_system:"solver_y_up_meters".into(),constrained_root_state_authority:"not_reported".into(),bodies,control_program_sha256:None,control_segments:None};
        f.verify(vertices)?; Ok(f)
    }
    pub fn verify(&self,vertices:&[u8])->Result<()> {
        verify_vertices(vertices)?;
        let control_valid=match self.version {
            1=>self.constrained_root_state_authority=="not_reported"&&self.control_program_sha256.is_none()&&self.control_segments.is_none(),
            2=>self.constrained_root_state_authority=="native_fem_control_segments_v2"&&self.control_program_sha256.as_ref().is_some_and(|h|digest(h))&&self.control_segments.as_ref().is_some_and(|rows|!rows.is_empty()),
            _=>false,
        };
        if !control_valid||self.kind!=KIND||self.measurement!=MEASUREMENT||self.coordinate_system!="solver_y_up_meters" {return Err("unsupported FEM pose contract".into());}
        if let Some(rows)=&self.control_segments {
            let mut previous="";
            for r in rows {
                use crate::fem_control_events::{Mode,Binding};
                let valid=|m,b|matches!((m,b),(Mode::Free,None)|(Mode::Static,Some(Binding::CurrentHold|Binding::Absolute))|(Mode::Kinematic,Some(Binding::CurrentFollow|Binding::Absolute)));
                if !uuid(&r.body_uuid)||r.body_uuid.as_str()<=previous||!self.bodies.iter().any(|b|b.body_uuid==r.body_uuid)||!valid(r.segment_mode,r.segment_pose_binding)||!valid(r.accepted_end_mode,r.accepted_end_pose_binding)||r.segment_epoch!=r.segment_event_cursor||r.accepted_end_epoch!=r.accepted_end_event_cursor||r.accepted_end_event_cursor<r.segment_event_cursor||r.accepted_end_event_cursor-r.segment_event_cursor>1{return Err("invalid FEM control segment state".into());}previous=&r.body_uuid;
            }
        }
        if !uuid(&self.scene_uuid)||![&self.scene_inputs_sha256,&self.fem_payload_sha256,&self.vertex_frame_sha256].iter().all(|s|digest(s)){return Err("invalid FEM pose identity".into());}
        if self.vertex_frame_byte_length!=vertices.len() as u64||self.vertex_frame_sha256!=hash(vertices){return Err("vertex frame hash/length mismatch".into());}
        let(t,a,b)=(self.frame_time_seconds,self.accepted_bracket_start_seconds,self.accepted_solver_time_seconds);
        if ![t,a,b].iter().all(|v|v.is_finite())||a<0.||t<a||t>b{return Err("invalid accepted time bracket".into());}
        match self.source {
            crate::pose_output::Source::InitialFrame if self.frame_id==0&&self.accepted_steps==0&&t==0.&&a==0.&&b==0.=>{},
            crate::pose_output::Source::AcceptedEndpoint if self.frame_id>0&&self.accepted_steps>0&&t==b&&b>a=>{},
            crate::pose_output::Source::InterpolatedAcceptedBracket if self.frame_id>0&&self.accepted_steps>0&&t<b&&b>a=>{},
            _=>return Err("source/acceptance metadata mismatch".into()),
        }
        let mut seen=HashSet::new();
        if self.bodies.is_empty(){return Err("empty FEM material map".into());}
        for p in &self.bodies {
            if !uuid(&p.body_uuid)||!seen.insert(&p.body_uuid)||p.body_type!="FEM_TET_MATERIAL"||!p.centroid_m.iter().all(|v|v.is_finite())||p.vertex_indices.iter().any(|i|*i as usize>=vertices.len()/12)||p.vertex_indices.iter().copied().collect::<HashSet<_>>().len()!=4{return Err("invalid material identity/geometry".into());}
            rotation(&p.rotation_column_major)?;
        } Ok(())
    }
}
fn verify_vertices(vertices:&[u8])->Result<()> {
    if vertices.is_empty()||vertices.len()%12!=0||vertices.chunks_exact(4).any(|c|!f32::from_le_bytes(c.try_into().unwrap()).is_finite()){return Err("invalid accepted vertex frame".into());} Ok(())
}
pub fn encode(f:&PoseFrame,vertices:&[u8])->Result<Vec<u8>> {f.verify(vertices)?;let mut b=Vec::new();ciborium::into_writer(f,&mut b).map_err(|e|e.to_string())?;Ok(b)}
pub fn decode(bytes:&[u8],vertices:&[u8],frame_id:u32,scene_uuid:&str,scene_hash:&str,payload_hash:&str,bindings:&[MaterialBinding])->Result<PoseFrame> {
    let mut c=std::io::Cursor::new(bytes);let f:PoseFrame=ciborium::from_reader(&mut c).map_err(|e|e.to_string())?;
    if c.position()!=bytes.len() as u64{return Err("trailing FEM pose data".into());} f.verify(vertices)?;
    if f.frame_id!=frame_id||f.scene_uuid!=scene_uuid||f.scene_inputs_sha256!=scene_hash||f.fem_payload_sha256!=payload_hash{return Err("FEM frame/scene/payload identity mismatch".into());}
    let expected=bindings.iter().map(|b|measure(b,vertices)).collect::<Result<Vec<_>>>()?;
    if expected!=f.bodies{return Err("material measurement differs from sealed binding/vertices".into());} Ok(f)
}
pub fn publish(output:&Path,f:&PoseFrame,vertices:&[u8])->Result<()> {
    if std::fs::read(output.join(format!("vert_{}.bin",f.frame_id))).map_err(|e|e.to_string())?!=vertices{return Err("published vertex generation differs".into());}
    let bytes=encode(f,vertices)?;let path=output.join(format!("fem_pose_{}.cbor",f.frame_id));
    if path.exists(){return if std::fs::read(path).map_err(|e|e.to_string())?==bytes{Ok(())}else{Err("conflicting FEM pose generation".into())};}
    let tmp=output.join(format!("fem_pose_{}.cbor.tmp",f.frame_id));
    let mut file=std::fs::OpenOptions::new().create_new(true).write(true).open(&tmp).map_err(|e|e.to_string())?;
    let done=(||{file.write_all(&bytes)?;file.sync_all()})().map_err(|e|e.to_string());drop(file);
    if let Err(e)=done {let _=std::fs::remove_file(&tmp);return Err(e);}
    if let Err(e)=std::fs::rename(&tmp,path){let _=std::fs::remove_file(&tmp);return Err(e.to_string());} Ok(())
}
