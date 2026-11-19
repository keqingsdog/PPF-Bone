//! Separate restoring V1 POD. No legacy record is modified.
//! Numeric enums: domain SIGNED=0 ODD=1 RADIAL=2; tail LINEAR=0 HOLD=1.
//! binding kind rigid angular=0,FEM angular=1,rigid anchor=2,FEM anchor=3,stitch=4,cross-link distance=5.
//! channel twist=0,swing_y=1,swing_z=2,anchor/stitch=3.
#[repr(C)]
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct CurveDescriptor {
    pub node_offset:u32, pub node_count:u32, pub domain:u32, pub tail:u32,
    pub reserved:[u32;4],
}
#[repr(C)]
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct CurveNode { pub x:f32, pub effort:f32, pub area:f32, pub right_slope:f32 }
#[repr(C)]
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct CurveBinding {
    pub kind:u32, pub target_index:u32, pub channel:u32, pub table_index:u32,
    pub history_index:u32, pub receipt_index:u32, pub damping:f32, pub epsilon_m:f32,
    pub rest_length_m:f32, pub reserved:[u32;3],
}
pub use crate::checkpoint_values::CurveHistory;
/// Dispatch field specification: arrays are read-only except capture's history output.
/// Driver argument wrapper must specify these exact element types; pointer representation
/// follows the existing generated backend mechanism, not embedded host pointers in POD.
pub const DISPATCH_FIELDS:&str="descriptors:CurveDescriptor[];nodes:CurveNode[];bindings:CurveBinding[];history:CurveHistory[];binding_count:u32;physical_dt:f32";
// Kind-domain semantics are an ABI identity input even when POD bytes stay fixed.
pub const LAYOUT_SPEC:&str="CurveBinding.kind5-direction-mask-v1:reserved0-marker1-tension2-compression4-q0-tension-branch-projected-continuous-damping;CurveBinding.kind-domain:RigidAngular0,FemAngular1,RigidAnchor2,FemAnchor3,Stitch4,CrossLinkDistance5;CurveDescriptor:size32:align4:node_offset@0:u32,node_count@4:u32,domain@8:u32,tail@12:u32,reserved@16:u32[4];CurveNode:size16:align4:x@0:f32,effort@4:f32,area@8:f32,right_slope@12:f32;CurveBinding:size48:align4:kind@0:u32,target_index@4:u32,channel@8:u32,table_index@12:u32,history_index@16:u32,receipt_index@20:u32,damping@24:f32,epsilon_m@28:f32,rest_length_m@32:f32,reserved@36:u32[3];CurveHistory:size32:align8:d0@0:f32[3],q0@12:f32,epoch@16:u64,initialized@24:u32,reserved@28:u32";
/// SI sidecar remains sealed. Create per-binding tables at installation so a
/// table shared by angular and length coordinates receives each unit transform
/// exactly once. E_native=s^5 E_SI, physical time is unchanged.
pub fn solver_space(tables:&crate::restoring_curve::PackedTables,bindings:&[CurveBinding],s:f64)->Result<(crate::restoring_curve::PackedTables,Vec<CurveBinding>),String> {
    if !s.is_finite()||s<=0. {return Err("RESTORING_ARITHMETIC_FAULT: world scaling".into());}
    let narrow=|x:f64|->Result<f32,String>{let y=x as f32;if !x.is_finite()||!y.is_finite()||(x!=0.&&y==0.){Err("RESTORING_ARITHMETIC_FAULT: solver-space conversion".into())}else{Ok(y)}};
    let mut out=crate::restoring_curve::PackedTables{descriptors:Vec::new(),nodes:Vec::new()};let mut rows=Vec::with_capacity(bindings.len());
    for source in bindings {
        let mut b=*source;let d=*tables.descriptors.get(b.table_index as usize).ok_or("RESTORING_TARGET_INVALID: table index")?;
        let end=d.node_offset.checked_add(d.node_count).ok_or("RESTORING_TARGET_INVALID: table range")?;
        let ns=tables.nodes.get(d.node_offset as usize..end as usize).ok_or("RESTORING_TARGET_INVALID: table nodes")?;
        let angular=b.kind<=1;let sx=if angular{1.}else{s};let sf=if angular{s.powi(5)}else{s.powi(4)};let sk=if angular{s.powi(5)}else{s.powi(3)};
        b.table_index=u32::try_from(out.descriptors.len()).map_err(|_|"RESTORING_ARITHMETIC_FAULT: descriptor count")?;
        let mut copy=d;copy.node_offset=u32::try_from(out.nodes.len()).map_err(|_|"RESTORING_ARITHMETIC_FAULT: node count")?;
        for n in ns {out.nodes.push(CurveNode{x:narrow(n.x as f64*sx)?,effort:narrow(n.effort as f64*sf)?,area:narrow(n.area as f64*s.powi(5))?,right_slope:narrow(n.right_slope as f64*sk)?});}
        b.damping=narrow(b.damping as f64*sk)?;if !angular{b.epsilon_m=narrow(b.epsilon_m as f64*s)?;b.rest_length_m=narrow(b.rest_length_m as f64*s)?;}
        out.descriptors.push(copy);rows.push(b);
    }Ok((out,rows))
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn layouts() {
        assert_eq!((std::mem::size_of::<CurveDescriptor>(),std::mem::align_of::<CurveDescriptor>()),(32,4));
        assert_eq!((std::mem::size_of::<CurveNode>(),std::mem::align_of::<CurveNode>()),(16,4));
        assert_eq!((std::mem::size_of::<CurveBinding>(),std::mem::align_of::<CurveBinding>()),(48,4));
        assert_eq!((std::mem::size_of::<CurveHistory>(),std::mem::align_of::<CurveHistory>()),(32,8));
        assert_eq!(std::mem::offset_of!(CurveHistory,epoch),16);
        assert_eq!(std::mem::offset_of!(CurveBinding,damping),24);
    }
    fn table()->crate::restoring_curve::PackedTables {
        crate::restoring_curve::PackedTables{descriptors:vec![CurveDescriptor{node_count:2,..Default::default()}],nodes:vec![CurveNode::default(),CurveNode{x:1.,effort:2.,area:1.,right_slope:2.}]}
    }
    #[test] fn shared_table_has_independent_angle_and_length_unit_transforms() {
        let bindings=[CurveBinding{kind:0,damping:3.,..Default::default()},CurveBinding{kind:4,damping:3.,epsilon_m:0.1,rest_length_m:0.2,..Default::default()}];
        let (tables,rows)=solver_space(&table(),&bindings,2.).unwrap();
        assert_eq!(tables.descriptors.len(),2);assert_eq!(rows[1].table_index,1);
        assert_eq!(tables.nodes[1],CurveNode{x:1.,effort:64.,area:32.,right_slope:64.});
        assert_eq!(tables.nodes[3],CurveNode{x:2.,effort:32.,area:32.,right_slope:16.});
        assert_eq!(rows[0].damping,96.);assert_eq!(rows[1].damping,24.);
        assert_eq!(rows[1].epsilon_m,0.2);assert_eq!(rows[1].rest_length_m,0.4);
        // s^5 E_SI, angle g_native=s^5g_SI, length g_native=s^4g_SI.
        assert_eq!(tables.nodes[1].area,tables.nodes[3].area);
    }
    #[test] fn unit_scale_keeps_fp32_values_and_rejects_nonrepresentable_scale() {
        let input=table();let bindings=[CurveBinding{kind:4,damping:3.,epsilon_m:0.1,rest_length_m:0.2,..Default::default()}];
        let (output,rows)=solver_space(&input,&bindings,1.).unwrap();
        assert_eq!(output.nodes,input.nodes);assert_eq!(rows,bindings);
        assert!(solver_space(&input,&bindings,0.).is_err());
        assert!(solver_space(&input,&bindings,1e-30).is_err());
        assert!(solver_space(&input,&bindings,1e30).is_err());
    }
}
