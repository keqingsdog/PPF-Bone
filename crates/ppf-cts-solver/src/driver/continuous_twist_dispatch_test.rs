//! Test contract for actual angular dispatch; adapter must call production
//! history/Device APIs. Deliberately no #[test] until the adapter exists: an
//! oracle-only pass must not be mistaken for functional coverage.

#[allow(dead_code)]
#[derive(Debug)]
struct AngularReadback { theta: f64, gradient: f64, energy: f64, hv: f64, fault: u32 }

#[allow(dead_code)]
trait ActualDispatch {
    type Error: std::fmt::Debug;
    // Implement using accepted history and real device buffer uploads. The
    // angle describes the physical rotation; the adapter must not unwrap it.
    fn begin(&mut self) -> Result<(), Self::Error>;
    fn evaluate(&mut self, angle: f64) -> Result<AngularReadback, Self::Error>;
    fn commit(&mut self) -> Result<(), Self::Error>;
    fn abort(&mut self) -> Result<(), Self::Error>;
}

#[allow(dead_code)]
fn close(label: &str, actual: f64, expected: f64, abs: f64, rel: f64) {
    assert!(actual.is_finite() && (actual - expected).abs() <= abs + rel * expected.abs(),
        "{label}: actual={actual} expected={expected}");
}

/// Pure x rotation, fixed parent, identity local frames and rest, no damping;
/// direct gradient is +k*theta (the physical restoring torque is its negative).
/// Consecutive poses are below pi apart, including both branch crossings.
#[allow(dead_code)]
fn assert_sequence<D: ActualDispatch>(device: &mut D, sign: f64, k: f64) {
    let degrees = [0., 90., 179., 181., 270., 359., 360., 450., 540., 630., 719., 720., 630., 540., 450., 360., 270., 180., 90., 0.];
    for d in degrees {
        let theta = sign * d * std::f64::consts::PI / 180.;
        device.begin().unwrap();
        let got = device.evaluate(theta).unwrap();
        assert_eq!(got.fault, 0, "fault at {d} degrees");
        close("theta", got.theta, theta, 3e-4, 2e-5);
        close("gradient", got.gradient, k * theta, 2e-3, 2e-4);
        close("energy", got.energy, 0.5 * k * theta * theta, 2e-3, 3e-4);
        // Pure x, unit tangent, no damping: J^T H J = k.
        close("Gauss-Newton Hv", got.hv, k, 2e-3, 2e-4);
        device.commit().unwrap();
    }
}

/// Evaluations in one trial do not commit; aborted candidates must leave the
/// accepted 179-degree branch intact for the 181-degree accepted candidate.
#[allow(dead_code)]
fn assert_abort_and_energy_fd<D: ActualDispatch>(device: &mut D, k: f64) {
    let rad = std::f64::consts::PI / 180.;
    device.begin().unwrap();
    device.evaluate(179. * rad).unwrap();
    device.commit().unwrap();
    device.begin().unwrap();
    for _ in 0..3 { device.evaluate(270. * rad).unwrap(); }
    device.abort().unwrap();
    device.begin().unwrap();
    let angle = 181. * rad;
    let eps = 0.001;
    let minus = device.evaluate(angle - eps).unwrap();
    let plus = device.evaluate(angle + eps).unwrap();
    let middle = device.evaluate(angle).unwrap();
    for sample in [&minus, &plus, &middle] { assert_eq!(sample.fault, 0); }
    close("after abort theta", middle.theta, angle, 3e-4, 2e-5);
    close("energy finite difference", (plus.energy - minus.energy) / (2. * eps), middle.gradient, 0.02, 0.003);
    close("after abort gradient", middle.gradient, k * angle, 2e-3, 2e-4);
    device.commit().unwrap();
}

// M1 only: explicit valid context, actual restoring angular kernel. This does
// not establish that production constructs the context or persists history.
#[test]
fn explicit_context_host_dispatch() {
    use ppf_cts_compute::{AllocLabel, Buffer, Device};
    use crate::continuous_twist::AngularTwistEvalContext;
    use crate::data::PdrdRigidState;
    use crate::joint_angular_control_gpu::AngularControlGpu;
    use super::kernels::JointRestoringAngularApplyArgs;
    let mut device = super::launch::host_device();
    fn upload<T: ppf_cts_compute::Pod>(d: &mut impl Device, values: &[T], label: &'static str) -> Buffer<T> {
        let mut b = Buffer::<T>::default();
        b.size(d, values.len(), AllocLabel(label)).unwrap();
        b.write(d, 0, values).unwrap(); b
    }
    let identity = [1.,0.,0.,0.,1.,0.,0.,0.,1.];
    let row = AngularControlGpu { body_a: u32::MAX, body_b: 0, enabled: 1, coordinate_model: 1,
        frame_a_wxyz:[1.,0.,0.,0.], frame_b_wxyz:[1.,0.,0.,0.], rest_relative_wxyz:[1.,0.,0.,0.],
        stiffness_nm_per_rad:[2.,0.,0.], damping_nms_per_rad:[0.;3] };
    let records = upload(&mut device, &[row], "twist.test.records");
    let mut state = upload(&mut device, &[PdrdRigidState::default()], "twist.test.state");
    let reference = upload(&mut device, &[PdrdRigidState { rotation: identity, ..Default::default() }], "twist.test.reference");
    let mut contexts = upload(&mut device, &[AngularTwistEvalContext::default()], "twist.test.context");
    let input = upload(&mut device, &[0.,0.,0.,1.,0.,0.], "twist.test.input");
    let mut output = upload(&mut device, &[0f32;6], "twist.test.output");
    let mut fault = upload(&mut device, &[0u32], "twist.test.fault");
    let dummy = upload(&mut device, &[0u64;8], "twist.test.dummy");
    for degrees in [-720f32,-360.,-181.,-179.,179.,181.,360.,720.] {
        let theta = degrees.to_radians(); let (s,c) = theta.sin_cos();
        state.write(&mut device, 0, &[PdrdRigidState { rotation:[1.,0.,0.,0.,c,s,0.,-s,c], ..Default::default() }]).unwrap();
        contexts.write(&mut device, 0, &[AngularTwistEvalContext { current_anchor:theta,reference_anchor:0.,continuous:1,initialized:1 }]).unwrap();
        for action in [0u32,1,4] {
            output.write(&mut device,0,&[0.;6]).unwrap(); fault.write(&mut device,0,&[0]).unwrap();
            let args=JointRestoringAngularApplyArgs {records:records.handle(),twist_context:contexts.handle(),material_spin_context:dummy.handle(),record_count:1,count:1,state:state.handle(),reference:reference.handle(),bodies:1,base:0,dim:6,dt:1.,action,input:input.handle(),output:output.handle(),fault:fault.handle(),descriptors:dummy.handle(),nodes:dummy.handle(),bindings:dummy.handle(),history:dummy.handle(),descriptor_count:0,node_count:0,binding_count:0,history_count:0,observations:dummy.handle(),seam_arena_count:0};
            unsafe { device.launch("restoring.angular",&args,1).unwrap(); }
            let mut f=[0]; fault.read(&mut device,0,&mut f).unwrap(); assert_eq!(f[0],0,"degree {degrees} action {action}");
            let mut out=[0.;6]; output.read(&mut device,0,&mut out).unwrap();
            let (actual,expected)=match action {0=>(out[3],2.*theta),1=>(out[3],2.),_=>(out[0],theta*theta)};
            close("actual dispatch",actual as f64,expected as f64,0.003,0.0003);
        }
    }
}


// Actual three-pass production device fit. The anisotropic box realizes the
// captured covariance from M2; the expected rotation comes from an independent
// f64 SVD rather than the production refinement helper.
#[test]
fn continuous_selected_real_fit_matches_svd_and_legacy_stays_unchanged() {
    use ppf_cts_compute::{AllocLabel, Buffer, Device};
    use crate::data::{PdrdBodyProp, PdrdRigidState, VertexProp};
    fn upload<T: ppf_cts_compute::Pod>(d:&mut impl Device,v:&[T],label:&'static str)->Buffer<T>{
        let mut b=Buffer::<T>::default();b.size(d,v.len(),AllocLabel(label)).unwrap();b.write(d,0,v).unwrap();b
    }
    let mut device=super::launch::backend();
    let c=nalgebra::Matrix3::<f64>::from_column_slice(&[0.00759884234658572,-0.26077853753011754,3.030581983991527,-0.6842889112238572,-0.8821081266793517,-2.3001813023608153,3.8696842895315875,-0.6452956339381852,10.575553560566526]);
    let svd=c.svd(true,true); let expected=svd.u.unwrap()*svd.v_t.unwrap();
    assert!(expected.determinant()>0.0);
    let transform=c*nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0/8.0,1.0/32.0,1.0/72.0));
    let mut rest=Vec::<f32>::new();let mut positions=Vec::<f32>::new();
    for x in [-1.0,1.0] {for y in [-2.0,2.0] {for z in [-3.0,3.0] {
        let v=nalgebra::Vector3::new(x,y,z); let q=transform*v;
        rest.extend(v.iter().map(|x|*x as f32));positions.extend(q.iter().map(|x|*x as f32));
    }}}
    let vertices:Vec<VertexProp>=(0..8).map(|_|VertexProp{pdrd_body_index:1,..Default::default()}).collect();
    let body=PdrdBodyProp{vertex_count:8,mass_per_vertex:1.,rest_gram_inv:nalgebra::Matrix3::identity(),..Default::default()};
    let list=upload(&mut device,&(0..8u32).collect::<Vec<_>>(),"fit.test.list");
    let props=upload(&mut device,&vertices,"fit.test.props");let bodies=upload(&mut device,&[body],"fit.test.body");
    let rest=upload(&mut device,&rest,"fit.test.rest");let positions=upload(&mut device,&positions,"fit.test.pos");
    let scratch=upload(&mut device,&[0f32;12],"fit.test.scratch");let state=upload(&mut device,&[PdrdRigidState::default()],"fit.test.state");
    let map=super::rigid_map::RigidMap{n_bodies:1,..Default::default()};let mut staged=super::rigid_map::Staged::default();
    fn fit<D:Device>(d:&mut D,map:&super::rigid_map::RigidMap,staged:&mut super::rigid_map::Staged,list:&Buffer<u32>,props:&Buffer<VertexProp>,body:&Buffer<PdrdBodyProp>,rest:&Buffer<f32>,pos:&Buffer<f32>,scratch:&Buffer<f32>,state:&Buffer<PdrdRigidState>)->PdrdRigidState{
        let mut reduction=super::pdrd::Reduction{map,staged};unsafe{reduction.fit(d,list.handle(),props.handle(),body.handle(),rest.handle(),pos.handle(),scratch.handle(),state.handle(),8,&[0.;12]).unwrap();}
        let mut result=[PdrdRigidState::default()];state.read(d,0,&mut result).unwrap();result[0]
    }
    let old=fit(&mut device,&map,&mut staged,&list,&props,&bodies,&rest,&positions,&scratch,&state);
    staged.continuous_fit_mask=vec![true];
    let corrected=fit(&mut device,&map,&mut staged,&list,&props,&bodies,&rest,&positions,&scratch,&state);
    for i in 0..9 {close("selected device fit",corrected.rotation[i] as f64,expected.as_slice()[i],0.0003,0.0001);}
    staged.continuous_fit_mask.clear();
    let old_again=fit(&mut device,&map,&mut staged,&list,&props,&bodies,&rest,&positions,&scratch,&state);
    assert_eq!(old.rotation,old_again.rotation,"legacy fit changed after selected-mode fit");
}
