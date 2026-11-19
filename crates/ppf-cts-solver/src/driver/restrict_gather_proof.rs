use super::rigid_map::{RigidMap, RIGID_UNSET};
use crate::data::PdrdBodyProp;

// Checked once, before dispatch: the lists must represent exactly the same
// partition as the authoritative per-vertex map used by the original row.
pub fn membership(map: &RigidMap, bodies: &[PdrdBodyProp], list: &[u32]) -> bool {
    if bodies.len() != map.n_bodies || map.vbody.len() != map.nrow ||
        map.n_bodies == 0 || map.dim > u32::MAX as usize || map.nrow > (u32::MAX / 3) as usize ||
        map.nrow.checked_add(map.n_bodies).and_then(|n| u32::try_from(n).ok()).is_none() {
        return false;
    }
    if map.cloth_off.len() != map.nrow || map.n_cloth.checked_mul(3) != Some(map.body_base) ||
        map.n_bodies.checked_mul(6).and_then(|n| map.body_base.checked_add(n)) != Some(map.dim) {
        return false;
    }
    let mut cloth = 0usize;
    for (v, &body) in map.vbody.iter().enumerate() {
        if body == RIGID_UNSET {
            if cloth.checked_mul(3) != Some(map.cloth_off[v] as usize) { return false; }
            cloth += 1;
        }
    }
    if cloth != map.n_cloth { return false; }
    let mut seen = Vec::<bool>::new();
    if seen.try_reserve_exact(map.nrow).is_err() { return false; }
    seen.resize(map.nrow, false);
    for (b, body) in bodies.iter().enumerate() {
        let start = body.vertex_start as usize;
        let Some(end) = start.checked_add(body.vertex_count as usize) else { return false; };
        let Some(vertices) = list.get(start..end) else { return false; };
        for &v in vertices {
            let v = v as usize;
            if v >= map.nrow || seen[v] || map.vbody[v] != b as u32 { return false; }
            seen[v] = true;
        }
    }
    map.vbody.iter().enumerate().all(|(v, &b)| if b == RIGID_UNSET { !seen[v] } else { seen[v] && (b as usize) < map.n_bodies })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn validates_partition_and_empty_body() {
        let mut a=PdrdBodyProp::default();a.vertex_start=0;a.vertex_count=2;
        let mut empty=PdrdBodyProp::default();empty.vertex_start=2;empty.vertex_count=0;
        let m=RigidMap{nrow:3,n_bodies:2,n_cloth:1,body_base:3,dim:15,vbody:vec![0,RIGID_UNSET,0],cloth_off:vec![0,0,0],..Default::default()};
        assert!(membership(&m,&[a,empty],&[2,0]));
        assert!(!membership(&m,&[a,empty],&[0,0]));
        assert!(!membership(&m,&[a,empty],&[0,1]));
        assert!(!membership(&m,&[a,empty],&[0,3]));
        assert!(!membership(&m,&[a,empty],&[0]));
        let mut missing=a;missing.vertex_count=1;assert!(!membership(&m,&[missing,empty],&[0,2]));
    }
}

#[cfg(test)]
mod device_oracle {
    use super::*;
    use ppf_cts_compute::{AllocLabel, Buffer, Device};
    use super::super::kernels::{PdrdRestrictGatherArgs, PdrdRestrictRowArgs};
    #[test]
    fn actual_host_kernel_matches_original_with_empty_body() {
        let mut d=super::super::launch::host_device();
        let mut body0=PdrdBodyProp::default();body0.vertex_start=0;body0.vertex_count=2;
        let mut body1=PdrdBodyProp::default();body1.vertex_start=2;body1.vertex_count=1;
        let mut body2=PdrdBodyProp::default();body2.vertex_start=3;body2.vertex_count=0;
        let owners=[u32::MAX,0,0,1,u32::MAX];let offsets=[0u32,0,0,0,3];let list=[2u32,1,3];
        let positions=[0.0f32,0.,0.,1.,2.,3.,-2.,1.,0.,0.,2.,-1.,0.,0.,0.];
        let values=[1.0f32,-2.,3.,4.,5.,-6.,7.,-8.,9.,-1.,2.,-3.,10.,11.,12.];
        macro_rules! upload {($v:expr,$typ:ty) => {{let v=$v;let mut b=Buffer::<$typ>::none();b.size(&mut d,v.len(),AllocLabel("gather.oracle")).unwrap();b.write(&mut d,0,&v).unwrap();b}}}
        let vb=upload!(owners,u32);let co=upload!(offsets,u32);let vl=upload!(list,u32);
        let bp=upload!([body0,body1,body2],PdrdBodyProp);let rp=upload!(positions,f32);let fv=upload!(values,f32);
        let out0=upload!([0.0f32;24],f32);let out1=upload!([-123.0f32;24],f32);
        let old=PdrdRestrictRowArgs{vertex_body:vb.handle(),cloth_offset:co.handle(),rotated_rest:rp.handle(),full:fv.handle(),cloth_out:out0.handle(),reduced:out0.handle(),body_base:6,count:5,seam_arena_count:0};
        let new=PdrdRestrictGatherArgs{vertex_body:vb.handle(),cloth_offset:co.handle(),vert_list:vl.handle(),body_prop:bp.handle(),rotated_rest:rp.handle(),full:fv.handle(),reduced:out1.handle(),body_base:6,nrow:5,nbodies:3,count:8,seam_arena_count:0};
        unsafe { d.launch("oracle.original",&old,5).unwrap();d.launch("oracle.gather",&new,8).unwrap(); }
        let mut x=[0.0f32;24];let mut y=[0.0f32;24];out0.read(&mut d,0,&mut x).unwrap();out1.read(&mut d,0,&mut y).unwrap();
        assert_eq!(x,y);assert_eq!(&y[18..24],&[0.0;6]);
    }
}
