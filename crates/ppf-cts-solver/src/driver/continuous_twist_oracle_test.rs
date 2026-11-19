//! Private production fit/force checks against prescribed rotations and a
//! finite-difference energy oracle. No test creates a release fault hook.
use ppf_cts_compute::{AllocLabel,Buffer,Device};
use crate::data::{PdrdBodyProp,PdrdRigidState,VertexProp};
use crate::continuous_twist::AngularTwistEvalContext;
use crate::joint_angular_control_gpu::AngularControlGpu;
use super::kernels::JointRestoringAngularApplyArgs;

fn upload<T:ppf_cts_compute::Pod>(d:&mut impl Device,v:&[T],label:&'static str)->Buffer<T>{
    let mut b=Buffer::default();b.size(d,v.len(),AllocLabel(label)).unwrap();b.write(d,0,v).unwrap();b
}

#[test]
fn selected_fit_known_rotations_and_real_energy_derivative() {
    let mut device=super::launch::backend();
    // Non-diagonal anisotropic rest Gram: independent known R, not an SVD
    // computed by the same implementation that is under test.
    let l=nalgebra::Matrix3::new(1.0,0.3,0.0, 0.0,2.0,0.4, 0.0,0.0,3.0);
    let mut rest=Vec::new();for x in [-1.,1.] {for y in [-1.,1.] {for z in [-1.,1.] {
        let v=l*nalgebra::Vector3::new(x,y,z);rest.extend(v.iter().map(|x|*x as f32));
    }}}
    let list=upload(&mut device,&(0..8u32).collect::<Vec<_>>(),"oracle.list");
    let props=upload(&mut device,&vec![VertexProp{pdrd_body_index:1,..Default::default()};8],"oracle.props");
    let body=upload(&mut device,&[PdrdBodyProp{vertex_count:8,mass_per_vertex:1.,rest_gram_inv:nalgebra::Matrix3::identity(),..Default::default()}],"oracle.body");
    let restbuf=upload(&mut device,&rest,"oracle.rest");let mut pos=upload(&mut device,&vec![0f32;24],"oracle.pos");
    let scratch=upload(&mut device,&[0f32;12],"oracle.scratch");let state=upload(&mut device,&[PdrdRigidState::default()],"oracle.state");
    let reference=upload(&mut device,&[PdrdRigidState{rotation:[1.,0.,0.,0.,1.,0.,0.,0.,1.],..Default::default()}],"oracle.ref");
    let row=AngularControlGpu{body_a:u32::MAX,body_b:0,enabled:1,coordinate_model:1,frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],rest_relative_wxyz:[1.,0.,0.,0.],stiffness_nm_per_rad:[2.,0.,0.],damping_nms_per_rad:[0.;3]};
    let records=upload(&mut device,&[row],"oracle.records");let mut contexts=upload(&mut device,&[AngularTwistEvalContext::default()],"oracle.context");
    let input=upload(&mut device,&[0f32,0.,0.,1.,0.,0.],"oracle.input");let mut output=upload(&mut device,&[0f32;6],"oracle.output");let mut fault=upload(&mut device,&[0u32],"oracle.fault");let dummy=upload(&mut device,&[0u64;8],"oracle.dummy");
    let map=super::rigid_map::RigidMap{n_bodies:1,..Default::default()};let mut staged=super::rigid_map::Staged::default();staged.continuous_fit_mask=vec![true];
    let mut evaluate=|theta:f64|->(f64,f64,f64){
        let actual=theta.rem_euclid(std::f64::consts::TAU);let rotation=nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::x_axis(),actual);
        let mut vertices=Vec::new();for v in rest.chunks_exact(3) {let p=rotation*nalgebra::Vector3::new(v[0] as f64,v[1] as f64,v[2] as f64);vertices.extend(p.iter().map(|x|*x as f32));}
        pos.write(&mut device,0,&vertices).unwrap();
        let mut reduction=super::pdrd::Reduction{map:&map,staged:&mut staged};unsafe{reduction.fit(&mut device,list.handle(),props.handle(),body.handle(),restbuf.handle(),pos.handle(),scratch.handle(),state.handle(),8,&[0.;12]).unwrap();}
        let mut fit=[PdrdRigidState::default()];state.read(&mut device,0,&mut fit).unwrap();
        for i in 0..9 {assert!((fit[0].rotation[i] as f64-rotation.matrix().as_slice()[i]).abs()<2e-5,"known rotation angle={theta} index={i}");}
        contexts.write(&mut device,0,&[AngularTwistEvalContext{current_anchor:theta as f32,reference_anchor:0.,continuous:1,initialized:1}]).unwrap();
        let mut result=[0.;3];for (k,action) in [0u32,1,4].into_iter().enumerate(){
            output.write(&mut device,0,&[0.;6]).unwrap();fault.write(&mut device,0,&[0]).unwrap();
            let args=JointRestoringAngularApplyArgs{records:records.handle(),twist_context:contexts.handle(),material_spin_context:dummy.handle(),record_count:1,count:1,state:state.handle(),reference:reference.handle(),bodies:1,base:0,dim:6,dt:1.,action,input:input.handle(),output:output.handle(),fault:fault.handle(),descriptors:dummy.handle(),nodes:dummy.handle(),bindings:dummy.handle(),history:dummy.handle(),descriptor_count:0,node_count:0,binding_count:0,history_count:0,observations:dummy.handle(),seam_arena_count:0};
            unsafe{device.launch("oracle.angular",&args,1).unwrap();}let mut f=[0];fault.read(&mut device,0,&mut f).unwrap();assert_eq!(f[0],0);
            let mut out=[0f32;6];output.read(&mut device,0,&mut out).unwrap();result[k]=f64::from(out[if action==4{0}else{3}]);
        }
        (result[0],result[1],result[2])
    };
    for deg in [-181.,-179.,179.,181.] {evaluate(f64::to_radians(deg));}
    let theta=2.0*std::f64::consts::TAU+0.4;let eps=0.002;
    let minus=evaluate(theta-eps).2;let plus=evaluate(theta+eps).2;let (gradient,hv,_)=evaluate(theta);
    assert!(((plus-minus)/(2.0*eps)-gradient).abs()<0.03,"energy derivative vs gradient");
    assert!((hv-2.0).abs()<0.003,"independent GN J^T H J");
}
