//! Read-only measurement on a separate image of the published vertex frame.
//! No writes to positions, rigid_state, running rotations, locks or joint refs.
use ppf_cts_compute::{AllocLabel, Buffer, ReadbackBuffer};
use crate::data::PdrdRigidState;
use crate::pose_output::GeometryMeasurement;
use super::scene::Fatal;

#[derive(Default)]
pub struct Buffers {
    positions: Buffer<f32>, scratch: Buffer<f32>, fit: ReadbackBuffer<PdrdRigidState>,
    zero: Vec<f32>,
}
pub(crate) fn measure_frame_positions(frame_positions:&[f32], world_scaling:f32) -> Result<Vec<GeometryMeasurement>,String> {
    if !world_scaling.is_finite() || world_scaling<=0.0 { return Err("pose measurement invalid world scaling".into()); }
    let mut guard=super::SOLVER.lock().map_err(|_| "pose measurement poisoned solver".to_string())?;
    let state=guard.as_mut().ok_or("pose measurement before initialization")?;
    if frame_positions.len()!=state.sizes.vertices.checked_mul(3).ok_or("vertex count overflow")? {
        return Err("pose measurement needs the full saved vertex namespace".into());
    }
    let positions:Vec<f32>=frame_positions.iter().map(|v| *v*world_scaling).collect();
    if !positions.iter().all(|v|v.is_finite()) {return Err("pose measurement nonfinite input".into());}
    let bodies=state.rigid.n_bodies;
    if bodies==0 {return Err("pose measurement without rigid bodies".into());}
    // Retain the native expected counts before borrowing staged topology for fit.
    let expected:Vec<u32>=state.rigid_staged.body_prop.host().iter().map(|b|b.vertex_count).collect();
    if expected.len()!=bodies {return Err("pose measurement body metadata count differs".into());}
    let geometry_rest=super::roots::root_geometry::rest_flat(state.rigid_staged.rest_centered.host())?;
    let result=super::with_device(|device| {
        let own=&mut state.pose_measurement;
        own.positions.size(device,positions.len(),AllocLabel("pose_measurement.positions"))?;
        own.positions.write(device,0,&positions)?;
        own.scratch.size(device,12*bodies,AllocLabel("pose_measurement.fit_scratch"))?;
        own.fit.size(device,bodies,AllocLabel("pose_measurement.fit"))?;
        own.zero.resize(12*bodies,0.0);
        let vert_list=state.rigid_staged.vert_list.handle();
        let body_prop=state.rigid_staged.body_prop.handle();
        let rest=state.rigid_staged.rest_centered.handle();
        let body_vertices=u32::try_from(state.rigid_staged.vert_list.len()).map_err(|_|Fatal::invariant("pose body vertex count exceeds u32"))?;
        let mut reduction=super::pdrd::Reduction {map:&state.rigid,staged:&mut state.rigid_staged};
        // All output state and accumulators are dedicated to measurement.
        unsafe {reduction.fit(device,vert_list,state.prop_vertex.handle(),body_prop,rest,
            own.positions.handle(),own.scratch.handle(),own.fit.handle(),body_vertices,&own.zero)?;}
        own.fit.download(device)?;
        let mut measured=Vec::with_capacity(bodies);
        for (i,s) in own.fit.host().iter().enumerate() {
            if s.count!=expected[i] {return Err(Fatal::invariant("pose fit vertex count differs from native body"));}
            let body=state.rigid_staged.body_prop.host()[i];
            let geometry=super::roots::root_geometry::fit_body(frame_positions,&geometry_rest,
                state.rigid_staged.vert_list.host(),i as u32,body.vertex_start,body.vertex_count,f64::from(world_scaling),
                |v|state.prop_vertex.host().get(v as usize).map(|p|p.pdrd_body_index))
                .map_err(|e|Fatal::invariant(format!("actual published frame geometry: {e}")))?;
            // An interpolated output frame can be nonrigid. Report its nearest
            // proper measurement; do not impose root residual qualification.
            measured.push(GeometryMeasurement {center:geometry.center,rotation:geometry.rotation.map(|v|v as f32),count:geometry.count});
        }
        Ok(measured)
    });
    result.map_err(|e|format!("read-only native pose measurement failed: {}",e.detail))
}
