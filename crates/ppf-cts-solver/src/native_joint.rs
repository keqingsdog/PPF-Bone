//! Joint input handoff, shared contact policy and explicit ordinary-run receipt.
use crate::data::DataSet;
use crate::joint::OwnedJointScene;
/// The official link table is shared by barrier, CCD and final intersection.
pub unsafe fn filter_connected(data:&mut DataSet,owned:&OwnedJointScene)->usize {
    if !owned.joints.iter().any(|j| j.wire.enabled && !j.wire.collide_connected && j.parent_runtime_body_index!=crate::joint_gpu::WORLD_ENDPOINT) { return 0; }
    let n=data.vertex.curr.size as usize;
    let mut rows=vec![Vec::<u32>::new();n];
    if data.start_link.size>0 {
        assert_eq!(data.start_link.size as usize,n,"joint filter link row count");
        for (i,row) in rows.iter_mut().enumerate() {
            let a=*data.start_link.offset.add(i) as usize; let b=*data.start_link.offset.add(i+1) as usize;
            for k in a..b { row.push(*data.start_link.data.add(k)); }
        }
    }
    let props=std::slice::from_raw_parts(data.prop.pdrd_body.data,data.prop.pdrd_body.size as usize);
    let verts=std::slice::from_raw_parts(data.pdrd_vert_list.data,data.pdrd_vert_list.size as usize);
    let mut pairs=0;
    for j in &owned.joints {
        if !j.wire.enabled || j.wire.collide_connected || j.parent_runtime_body_index==crate::joint_gpu::WORLD_ENDPOINT { continue; }
        let a=&props[j.parent_runtime_body_index as usize]; let b=&props[j.child_runtime_body_index as usize];
        let aa=&verts[a.vertex_start as usize..(a.vertex_start+a.vertex_count) as usize];
        let bb=&verts[b.vertex_start as usize..(b.vertex_start+b.vertex_count) as usize];
        for &u in aa { for &v in bb { rows[u as usize].push(v); rows[v as usize].push(u); } } pairs+=1;
    }
    if pairs>0 {
        for row in &mut rows { row.sort_unstable(); row.dedup(); }
        let nnz:usize=rows.iter().map(Vec::len).sum(); assert!(nnz<=u32::MAX as usize,"joint link table exceeds u32");
        data.start_link=crate::cvecvec::CVecVec::from(rows.as_slice());
    } pairs
}
pub fn write_receipt(output:&str,owned:&OwnedJointScene,accepted:u64,time:f64,status:&str,filtered:usize) {
    let value=serde_json::json!({"schema":"ppf.native-joints.v2", "implementation":"release2026-09-27-neutral-v2",
        "status":status,"accepted_steps":accepted,"accepted_time":time,"scene_uuid":owned.scene_uuid,
        "scene_inputs_sha256":owned.scene_inputs_sha256,"payload_sha256":owned.payload_sha256,
        "joint_count":owned.joints.len(),"enabled_joint_count":owned.joints.iter().filter(|j|j.wire.enabled).count(),
        "connected_collision_pairs_filtered":filtered,"solve":"authoritative_ppf_reduced_pcg",
        "ball_anchor":true,"isotropic_angular_v2":true,"hard_joint":!owned.hard_records.is_empty(),"d6":owned.payload_version==7,
        "output_pose_pullback":false,"physical_acceptance":"not_assessed"});
    std::fs::write(std::path::Path::new(output).join("native_joint_receipt.json"),serde_json::to_vec_pretty(&value).unwrap()).expect("joint receipt write");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cvec::CVec;
    fn owned()->OwnedJointScene {
        let root=tempfile::tempdir().unwrap();std::fs::create_dir(root.path().join("bin")).unwrap();
        let payload:crate::joint::JointPayload=serde_json::from_value(serde_json::json!({
            "required_capability":"pdrd_ball_anchor_v1","scene_uuid":"11111111-1111-1111-1111-111111111111","scene_inputs_sha256":"",
            "bodies":[{"body_uuid":"22222222-2222-2222-2222-222222222222","runtime_body_index":0},{"body_uuid":"33333333-3333-3333-3333-333333333333","runtime_body_index":1}],
            "joints":[{"joint_uuid":"44444444-4444-4444-4444-444444444444","parent_body_uuid":"22222222-2222-2222-2222-222222222222","child_body_uuid":"33333333-3333-3333-3333-333333333333",
                "parent_local_frame":{"translation_m":[0,0,0],"quaternion_wxyz":[1,0,0,0]},"child_local_frame":{"translation_m":[0,0,0],"quaternion_wxyz":[1,0,0,0]},"stiffness_N_per_m":10,"enabled":true,"collide_connected":false}]
        })).unwrap();
        std::fs::write(root.path().join(crate::joint::JOINT_FILE),ppf_cts_formats::envelope::to_cbor_with_version(2,crate::joint::KIND,&payload).unwrap()).unwrap();
        crate::joint::ingest(root.path(),&[0.;46]).unwrap().unwrap()
    }
    #[test]
    fn connected_collision_links_preserve_existing_links_and_other_objects() {
        let owned=owned();
        let mut data:DataSet=unsafe {std::mem::zeroed()};
        data.vertex.curr=CVec::from(vec![crate::data::Vec3f::zeros();6].as_slice());
        let mut a=crate::data::PdrdBodyProp::default();a.vertex_start=0;a.vertex_count=2;
        let mut b=a;b.vertex_start=2;
        data.prop.pdrd_body=CVec::from([a,b].as_slice());data.pdrd_vert_list=CVec::from([0u32,1,2,3].as_slice());
        data.start_link=crate::cvecvec::CVecVec::from([vec![5u32],vec![],vec![],vec![],vec![],vec![0]].as_slice());
        assert_eq!(unsafe {filter_connected(&mut data,&owned)},1);
        let row=|i:usize| unsafe {let a=*data.start_link.offset.add(i) as usize;let b=*data.start_link.offset.add(i+1) as usize;std::slice::from_raw_parts(data.start_link.data.add(a),b-a)};
        assert_eq!(row(0),[2,3,5]);assert_eq!(row(1),[2,3]);assert_eq!(row(2),[0,1]);assert_eq!(row(3),[0,1]);assert!(row(4).is_empty());assert_eq!(row(5),[0]);
        let mut disabled=owned.clone();disabled.joints[0].wire.enabled=false;
        let mut empty:DataSet=unsafe {std::mem::zeroed()};
        assert_eq!(unsafe {filter_connected(&mut empty,&disabled)},0);assert_eq!(empty.start_link.size,0);
        disabled.joints[0].wire.enabled=true;disabled.joints[0].wire.collide_connected=true;
        assert_eq!(unsafe {filter_connected(&mut empty,&disabled)},0);assert_eq!(empty.start_link.size,0);
    }
}
