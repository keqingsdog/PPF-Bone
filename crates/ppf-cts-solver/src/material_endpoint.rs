//! Shared physical body/anchor validation. This module has no graph semantics.
use crate::{joint::{self,JointError},soft_joint::{BodyWire,bad,floats,bary,rotation},soft_joint_gpu::*};
use nalgebra::{Matrix3,Vector3};
use std::collections::{HashMap,HashSet};
pub(crate) fn validate_bodies(bodies:&[BodyWire],pos:&[Vector3<f64>],tet:&[u32],membership:&HashMap<String,Vec<u32>>,rigid_vertex:&[u32],base:&joint::OwnedJointScene)->Result<(HashMap<String,FemEndpointGpu>,HashSet<u32>),JointError> {
    let mut endpoints=HashMap::new(); let mut fem_support=HashSet::new();
    for (i,b) in bodies.iter().enumerate() {
        if joint::canonical_uuid(&b.body_uuid).as_deref()!=Some(b.body_uuid.as_str()) || endpoints.contains_key(&b.body_uuid) {return Err(bad("body_uuid","canonical unique body identity required"));}
        let e=if b.kind=="PDRD" {
            let id=b.runtime_body_index.ok_or_else(||bad("runtime_body_index","genuine PDRD index required"))?;
            if base.bodies.get(&b.body_uuid)!=Some(&id) || b.tet_index.is_some() || b.vertex_indices.is_some() || b.dm_inv_column_major.is_some() || b.center_bary.is_some() || b.frame_rest_column_major.is_some() {return Err(bad("PDRD","body must be a real sealed PDRD and carry no FEM fields"));}
            FemEndpointGpu {kind:FEM_PDRD,id,vertices:[0;4],dm_inv:[0.;9],anchor_bary:[0.;4],center_bary:[0.;4],rest_rotation:[1.,0.,0.,0.,1.,0.,0.,0.,1.],local_anchor:[0.;3]}
        } else if b.kind=="FEM_TET_MATERIAL" {
            if b.runtime_body_index.is_some() || base.bodies.contains_key(&b.body_uuid) {return Err(bad("FEM","FEM identity cannot own a PDRD body row"));}
            let ti=b.tet_index.ok_or_else(||bad("tet_index","required"))? as usize;
            let nodes=b.vertex_indices.ok_or_else(||bad("vertex_indices","required"))?;
            if ti>=tet.len()/4 || tet[4*ti..4*ti+4]!=nodes || nodes.iter().copied().collect::<HashSet<_>>().len()!=4 {return Err(bad("tet_index","must match actual native tet order and four distinct nodes"));}
            let members=membership.get(&b.body_uuid).ok_or_else(||bad("membership","FEM object absent from actual vertex map"))?;
            for v in nodes {
                // Sealed file uses 0=ordinary, 1-based PDRD indices. The native
                // builder converts that to MAX/zero-based only after decoding.
                if v as usize>=pos.len() || !members.contains(&v) || rigid_vertex[v as usize]!=0 {return Err(bad("membership","material support must be ordinary nodes in this actual object"));} fem_support.insert(v);
            }
            let inv=b.dm_inv_column_major.ok_or_else(||bad("dm_inv","required"))?;
            let im=Matrix3::from_column_slice(&inv);
            if !inv.iter().all(|x|x.is_finite()) {return Err(bad("dm_inv","nonfinite"));}
            let x=nodes.map(|v|pos[v as usize]);
            let dm=Matrix3::from_columns(&[x[1]-x[0],x[2]-x[0],x[3]-x[0]]);
            // Reference winding is a convention, not a validity test; det(F)<=0 is guarded
            // where F is formed. The 2e-4 inverse-consistency residual, finiteness and
            // singular refusals are unchanged. soft_pose_output already accepts signed Dm.
            let det=dm.determinant();
            if !det.is_finite() || det==0. || (dm*im-Matrix3::identity()).norm()>2e-4 {return Err(bad("dm_inv","must invert the actual rest tet within f32 input precision"));}
            let cb=b.center_bary.ok_or_else(||bad("center_bary","required"))?; bary(&cb)?;
            let rr=b.frame_rest_column_major.ok_or_else(||bad("frame_rest","required"))?; rotation(&rr)?;
            FemEndpointGpu {kind:FEM_TET,id:u32::try_from(i).map_err(|_|bad("bodies","u32 budget"))?,vertices:nodes,dm_inv:floats(&inv)?,anchor_bary:[0.;4],center_bary:floats(&cb)?,rest_rotation:floats(&rr)?,local_anchor:[0.;3]}
        } else {return Err(bad("kind","unknown body material endpoint"));};
        endpoints.insert(b.body_uuid.clone(),e);
    }
    Ok((endpoints,fem_support))
}
pub(crate) fn bind_anchor(mut e:FemEndpointGpu, weights:Option<[f64;4]>, f:&joint::LocalFrame,pos:&[Vector3<f64>])->Result<FemEndpointGpu,JointError> {
            e.local_anchor=floats(&f.translation_m)?;
            if e.kind==FEM_TET {
                let weights=weights.ok_or_else(||bad("anchor_bary","required for material endpoint"))?; bary(&weights)?;
                let nodes=e.vertices.map(|v|pos[v as usize]);
                let cb=e.center_bary.map(f64::from); let rr=Matrix3::from_column_slice(&e.rest_rotation.map(f64::from));
                let anchor=(0..4).fold(Vector3::zeros(),|sum,k|sum+nodes[k]*weights[k]);
                let center=(0..4).fold(Vector3::zeros(),|sum,k|sum+nodes[k]*cb[k]);
                let expected=center+rr*Vector3::from_column_slice(&f.translation_m);
                if (anchor-expected).norm()>2e-6 {return Err(bad("anchor_bary","does not reconstruct original material joint anchor"));}
                e.anchor_bary=floats(&weights)?;
            } else if weights.is_some() {return Err(bad("anchor_bary","non-FEM endpoint must not carry material weights"));}
            Ok(e)
}
