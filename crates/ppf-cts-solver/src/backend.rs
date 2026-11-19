// Fork modification: changed from the upstream baseline.
// File: backend.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::cvec::CVec;
use super::data::{
    Constraint, EdgeParam, FaceParam, HingeParam, IntersectionRecord, Mat2x2f, Mat3x3f,
    MaterialParamUpdate, RestShapeUpdate, StepResult, Vec3f, VertexParam,
    MAX_INTERSECTION_RECORDS,
};
use super::plastic_state::{PlasticKinds, PlasticState};
use super::{mesh::Mesh, DataSet, ParamSet, ProgramArgs, Scene, SimArgs};
use chrono::Local;
use log::*;
use na::{Matrix2x3, Matrix3xX};
use ppf_cts_formats::status::{crash_kind_from_step, Outcome, Phase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[cfg(all(test,feature="cpu"))]
#[path="backend_sampler_test.rs"]
mod backend_sampler_test;

// The C++ side picks the implementation, and build.rs flips the link:
// libsimbackend_cuda for the CUDA path, libsimbackend_metal for Metal.
// Every library exposes the identical extern "C" surface, so this
// declaration is unconditional.
extern "C" {
    fn advance(result: *mut StepResult);
    fn advance_retry(result: *mut StepResult, continuation: u64, rejection_budget: u32) -> u64;
    fn adaptive_retry_capture_seed();
    fn adaptive_retry_restore_seed();
    fn adaptive_retry_discard_seed();
    fn adaptive_retry_cancel(continuation: u64);
    fn fetch();
    fn fetch_inv_rest();
    fn fetch_rest_angles();
    fn fetch_dyn_counts(n_value: *mut u32, n_offset: *mut u32);
    fn fetch_dyn(index: *mut u32, value: *mut f32, offset: *mut u32);
    fn update_dyn(index: *const u32, offset: *const u32);
    fn update_constraint(constraint: *const Constraint);
    fn update_rest_shape(update: *const RestShapeUpdate);
    fn update_material_params(update: *const MaterialParamUpdate);
    fn initialize(data: *const DataSet, param: *const ParamSet) -> bool;
    fn override_velocity(indices: *const u32, count: u32, vx: f32, vy: f32, vz: f32, dt: f32);
    fn gather_current_positions(indices: *const u32, count: u32, out: *mut f32);
    #[allow(clippy::too_many_arguments)]
    fn override_angular_velocity(
        indices: *const u32,
        count: u32,
        wx: f32,
        wy: f32,
        wz: f32,
        cx: f32,
        cy: f32,
        cz: f32,
        dt: f32,
    );
    fn init_collision_windows(
        vert_dmap: *const u32,
        vert_count: u32,
        windows: *const f32,
        window_counts: *const u32,
        n_groups: u32,
    );
    fn refresh_collision_active(time: f32);
    fn fetch_intersection_records(out: *mut IntersectionRecord, max_count: u32) -> u32;
}

impl crate::driver::adaptive_controller::DtParameter for ParamSet {
    fn authored_dt(&self) -> f32 { self.dt }
    fn set_dt(&mut self, dt: f32) { self.dt = dt; }
}

pub(crate) fn adaptive_cg_retry_enabled() -> bool {
    std::env::var_os("PPF_ADAPTIVE_CG_RETRY").is_some_and(|v| v == "1")
}

/// Own the device-side previous-position snapshot for exactly one backend
/// logical step. Drop invokes the explicit device free even when host-side
/// preparation unwinds before the retry loop reaches its normal exit.
struct AdaptiveSeedSnapshotGuard {
    active: bool,
}

impl AdaptiveSeedSnapshotGuard {
    fn new() -> Self { Self { active: false } }

    unsafe fn capture_or_restore(&mut self) {
        if self.active {
            adaptive_retry_restore_seed();
        } else {
            adaptive_retry_capture_seed();
            self.active = true;
        }
    }

    unsafe fn finish(&mut self) {
        if self.active {
            adaptive_retry_discard_seed();
            self.active = false;
        }
    }
}

impl Drop for AdaptiveSeedSnapshotGuard {
    fn drop(&mut self) {
        unsafe { self.finish() };
    }
}

/// Fallible host-only seed management lets a retry preserve the original
/// material refusal when a later device operation fails.
struct HostSeedSnapshotGuard {active:bool}
impl HostSeedSnapshotGuard {
    fn new()->Self {Self{active:false}}
    fn capture_or_restore(&mut self)->Result<(),crate::driver::HostFatal>{
        if self.active {crate::driver::host_seed_restore()} else {crate::driver::host_seed_capture()?;self.active=true;Ok(())}
    }
    fn finish(&mut self)->Result<(),crate::driver::HostFatal>{
        if self.active {self.active=false;crate::driver::host_seed_discard()}else{Ok(())}
    }
}
impl Drop for HostSeedSnapshotGuard {fn drop(&mut self){if self.active {let _=self.finish();}}}

#[cfg(all(test,feature="cpu"))]
pub(crate) fn private_test_failed_guard_finish() {
    use crate::driver::test_device_fault::{Armed,Point};
    let mut guard=HostSeedSnapshotGuard::new();guard.capture_or_restore().unwrap();
    let armed=Armed::new(&[Point::SeedDiscardFree]);
    let err=guard.finish().unwrap_err();assert!(err.detail.contains("injected point 3"));
    drop(guard);assert_eq!(Armed::hits(Point::SeedDiscardFree),1);drop(armed);
}

#[cfg(test)]
#[derive(Debug)]
struct PrivatePreparedObservation { kind: &'static str, uploaded: bool, payload: Vec<u8> }
#[cfg(test)]
thread_local! {
    static PRIVATE_PREPARED_OBSERVATIONS: std::cell::RefCell<Vec<PrivatePreparedObservation>> = const { std::cell::RefCell::new(Vec::new()) };
    static PRIVATE_PREPARED_CAPTURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
fn private_observe<T: serde::Serialize>(kind: &'static str, uploaded: bool, payload: &T) {
    if !PRIVATE_PREPARED_CAPTURE.with(|enabled| enabled.get()) { return; }
    let payload = bincode::serialize(payload).expect("private preparation observation");
    PRIVATE_PREPARED_OBSERVATIONS.with(|items| items.borrow_mut().push(PrivatePreparedObservation { kind, uploaded, payload }));
}

#[cfg(all(test,feature="cpu"))]
pub(crate) fn private_observed_constraint_upload() -> bool {
    PRIVATE_PREPARED_OBSERVATIONS.with(|items|items.borrow().iter().any(|item|item.kind=="constraint_fix" && item.uploaded))
}

/// Restricted fixture adapter: 03 has no rest/material schedules in this
/// control-flow test. Other authoring combinations need separate fixtures.
#[cfg(test)]
pub(crate) fn private_prepare_03(
    scene: &Scene, start: f64, h: f32, playback: f32,
    plan: Option<crate::driver::HostPreparedPlan>,
) -> Result<Vec<(&'static str, bool, Vec<u8>)>, crate::driver::HostFatal> {
    PRIVATE_PREPARED_OBSERVATIONS.with(|items| items.borrow_mut().clear());
    PRIVATE_PREPARED_CAPTURE.with(|enabled| enabled.set(true));
    let integration_h = plan.map_or(h, |p| p.integration_h());
    let result = prepare_attempt_inputs(scene, start, h, playback, plan, integration_h, integration_h,
        false, true, false, false, &[], None);
    PRIVATE_PREPARED_CAPTURE.with(|enabled| enabled.set(false));
    result?;
    Ok(PRIVATE_PREPARED_OBSERVATIONS.with(|items| items.borrow_mut().drain(..)
        .map(|item| (item.kind, item.uploaded, item.payload)).collect()))
}

/// Prepare exactly the authored inputs for one host attempt. The caller owns
/// seed restoration, plan identity, the accepted clock and retry accounting.
#[allow(clippy::too_many_arguments)]
fn prepare_attempt_inputs(
    scene: &Scene, start_time: f64, requested_h: f32, playback: f32,
    prepared: Option<crate::driver::HostPreparedPlan>, integration_h: f32, history_h: f32,
    fem_events: bool, host_fem_retry: bool, collision_windows_enabled: bool, start_membership: bool,
    rest_shape_keyframes: &[RestShapeKeyframe], material_keyframes: Option<&[MaterialKeyframe]>,
) -> Result<(), crate::driver::HostFatal> {
            // Pin target = end-of-step time. The implicit integrator
            // pulls pinned vertices toward this constraint during the
            // step, so picking pin_at(state.time + dt) means the
            // post-advance positions match what frontend.FixedScene.
            // time(state.time + dt) computes, vertex for vertex. The
            // previous wiring used state.time (start of step) as the
            // target which left every pin one step behind by the end
            // of the step.
            
            let target_time = if let Some(plan)=prepared {plan.end_s()} else if fem_events {crate::driver::fem_control_endpoint(start_time,requested_h,playback).expect("FEM event endpoint").expect("installed FEM event program")}else{start_time + requested_h as f64};
            #[cfg(test)] private_observe("clock", false, &(start_time, target_time, integration_h));
            let mut constraint = scene.make_constraint_with_membership(target_time, start_time, start_membership);
            if fem_events {crate::driver::prepare_fem_control_constraint(&mut constraint,start_time,requested_h,playback).expect("FEM exact control constraint");}
            #[cfg(test)] private_observe("constraint_fix", false, &constraint.fix.as_slice());
            if host_fem_retry {unsafe{crate::driver::host_update_constraint(&constraint)}?;}else{unsafe { update_constraint(&constraint) }};
            #[cfg(test)] private_observe("constraint_fix", true, &constraint.fix.as_slice());

            // Stream the time-varying rest shape for this step's target time,
            // right after the pin constraint (same per-frame upload pattern).
            // The elastic kernels re-read inv_rest each Newton iteration, so
            // this drives each element's rest pose toward the captured
            // deformation while pull pins guide it and contact still resolves.
            if !rest_shape_keyframes.is_empty() {
                let rest_shape = interp_rest_shape(rest_shape_keyframes, target_time);
                #[cfg(test)] private_observe("rest", false, &(rest_shape.inv_rest2x2.as_slice(),rest_shape.inv_rest3x3.as_slice(),rest_shape.exclude_face.as_slice(),rest_shape.exclude_tet.as_slice()));
                if host_fem_retry {unsafe{crate::driver::host_update_rest_shape(&rest_shape)}?;}else{unsafe { update_rest_shape(&rest_shape) }};
                #[cfg(test)] private_observe("rest", true, &(rest_shape.inv_rest2x2.as_slice(),rest_shape.inv_rest3x3.as_slice(),rest_shape.exclude_face.as_slice(),rest_shape.exclude_tet.as_slice()));
            }

            // Stream this step's material tables, on the same schedule and for
            // the same reason as the rest shape: the energy kernels read the
            // param arrays fresh each Newton iteration, so overwriting them
            // here drives the material through the step.
            if let Some(keyframes) = material_keyframes {
                if !keyframes.is_empty() {
                    let (face, vertex, edge, hinge) =
                        interp_material_params(keyframes, target_time);
                    let update = MaterialParamUpdate {
                        face: CVec::from(&face[..]),
                        vertex: CVec::from(&vertex[..]),
                        edge: CVec::from(&edge[..]),
                        hinge: CVec::from(&hinge[..]),
                    };
                    #[cfg(test)] private_observe("material", false, &(&face, &vertex, &edge, &hinge));
                    if host_fem_retry {unsafe{crate::driver::host_update_material_params(&update)}?;}else{unsafe { update_material_params(&update) }};
                    #[cfg(test)] private_observe("material", true, &(&face, &vertex, &edge, &hinge));
                }
            }

            // Overrides are keyed to the scene clock, not the physical integration step.
            let event_window_end = target_time;
            for (indices, vx, vy, vz) in scene.get_velocity_overrides(start_time, event_window_end) {
                #[cfg(test)] private_observe("velocity", false, &(&indices,vx,vy,vz,integration_h,history_h));
                if host_fem_retry {unsafe{crate::driver::host_override_velocity(&indices,vx,vy,vz,history_h)}?;}else{unsafe {
                    override_velocity(indices.as_ptr(), indices.len() as u32, vx, vy, vz, history_h);
                }}
                #[cfg(test)] private_observe("velocity", true, &(&indices,vx,vy,vz,integration_h,history_h));
            }
            // Angular velocity overrides: the spin axis is resolved from the
            // body's LIVE geometry each firing (gather current positions, then
            // a principal-axis solve), so it tracks the simulated (rotated /
            // deformed) pose rather than a frame-0 pose. Runs AFTER the linear
            // loop so a keyframe carrying both yields a full rigid-velocity
            // overwrite: prev = curr - (v_lin + ω × (x - c)) * dt.
            for (indices, pca_index, speed) in
                scene.get_angular_velocity_overrides(start_time, event_window_end)
            {
                let count = indices.len();
                let mut positions = vec![0.0f32; count * 3];
                if host_fem_retry {unsafe{crate::driver::host_gather_current_positions(&indices,&mut positions)}?;}else{unsafe {
                    gather_current_positions(
                        indices.as_ptr(),
                        count as u32,
                        positions.as_mut_ptr(),
                    );
                }}
                if let Some((c, axis)) = resolve_principal_axis(&positions, pca_index as usize) {
                    let (wx, wy, wz) = (axis[0] * speed, axis[1] * speed, axis[2] * speed);
                    #[cfg(test)] private_observe("angular", false, &(&indices,wx,wy,wz,c,integration_h,history_h));
                    if host_fem_retry {unsafe{crate::driver::host_override_angular_velocity(&indices,wx,wy,wz,c[0],c[1],c[2],history_h)}?;}else{unsafe {
                        override_angular_velocity(
                            indices.as_ptr(),
                            count as u32,
                            wx,
                            wy,
                            wz,
                            c[0],
                            c[1],
                            c[2],
                            history_h,
                        );
                    }}
                    #[cfg(test)] private_observe("angular", true, &(&indices,wx,wy,wz,c,integration_h,history_h));
                }
            }
            // Fixed world-axis angular overrides (World X/Y/Z / Custom): ω is
            // a world-space vector given directly, so no principal-axis solve
            // is needed; only the live centroid is gathered for the ω × (x-c)
            // pivot.
            for (indices, omega) in
                scene.get_angular_velocity_world_overrides(start_time, event_window_end)
            {
                let count = indices.len();
                let mut positions = vec![0.0f32; count * 3];
                if host_fem_retry {unsafe{crate::driver::host_gather_current_positions(&indices,&mut positions)}?;}else{unsafe {
                    gather_current_positions(
                        indices.as_ptr(),
                        count as u32,
                        positions.as_mut_ptr(),
                    );
                }}
                if let Some(c) = centroid_of(&positions) {
                    #[cfg(test)] private_observe("angular", false, &(&indices,omega,c,integration_h,history_h));
                    if host_fem_retry {unsafe{crate::driver::host_override_angular_velocity(&indices,omega[0],omega[1],omega[2],c[0],c[1],c[2],history_h)}?;}else{unsafe {
                        override_angular_velocity(
                            indices.as_ptr(),
                            count as u32,
                            omega[0],
                            omega[1],
                            omega[2],
                            c[0],
                            c[1],
                            c[2],
                            history_h,
                        );
                    }}
                    #[cfg(test)] private_observe("angular", true, &(&indices,omega,c,integration_h,history_h));
                }
            }
            if host_fem_retry {if collision_windows_enabled {crate::driver::host_refresh_collision_active(start_time as f32)?;}}else{unsafe { refresh_collision_active(start_time as f32) }};
    Ok(())
}

/// One precomputed rest-shape keyframe: `(time, inv_rest2x2, inv_rest3x3,
/// exclude_face, exclude_tet)`. The exclude masks (1 = element is near-singular
/// at this keyframe and dropped from the energy) come from
/// `builder::compute_inv_rest`.
type RestShapeKeyframe = (f64, Vec<Mat2x2f>, Vec<Mat3x3f>, Vec<u8>, Vec<u8>);

/// Linearly interpolate a precomputed inverse-rest keyframe schedule to
/// `time`, component-wise on the matrices, and pack the result for upload via
/// `update_rest_shape`. Times outside the schedule clamp to the nearest end.
/// Endpoints reproduce `builder::compute_inv_rest` exactly (alpha 0). An
/// element is excluded if it is near-singular in *either* bracketing keyframe
/// (OR of the masks), so a element collapsing between two samples is dropped
/// for the whole interval (its interpolated inv_rest is unused while excluded).
fn interp_rest_shape(keyframes: &[RestShapeKeyframe], time: f64) -> RestShapeUpdate {
    debug_assert!(
        keyframes.windows(2).all(|w| w[0].0 < w[1].0),
        "rest-shape keyframe times must be strictly increasing"
    );
    let n = keyframes.len();
    let (i0, i1, alpha) = if time <= keyframes[0].0 {
        (0, 0, 0.0f32)
    } else if time >= keyframes[n - 1].0 {
        (n - 1, n - 1, 0.0f32)
    } else {
        // Keyframe times are strictly increasing (the producer sorts a set of
        // boundary times), so the bracketing pair is found with a binary search
        // instead of a linear scan. In this interior branch
        // keyframes[0].0 < time < keyframes[n-1].0, so partition_point returns
        // hi in 1..=n-1, keeping lo and keyframes[hi] in bounds.
        let hi = keyframes.partition_point(|k| k.0 < time);
        let lo = hi - 1;
        let (t0, t1) = (keyframes[lo].0, keyframes[hi].0);
        let a = if t1 > t0 {
            ((time - t0) / (t1 - t0)) as f32
        } else {
            0.0
        };
        (lo, hi, a)
    };
    let inv2: Vec<Mat2x2f> = keyframes[i0]
        .1
        .iter()
        .zip(keyframes[i1].1.iter())
        .map(|(x, y)| *x + (*y - *x) * alpha)
        .collect();
    let inv3: Vec<Mat3x3f> = keyframes[i0]
        .2
        .iter()
        .zip(keyframes[i1].2.iter())
        .map(|(x, y)| *x + (*y - *x) * alpha)
        .collect();
    let exclude_face: Vec<u8> = keyframes[i0]
        .3
        .iter()
        .zip(keyframes[i1].3.iter())
        .map(|(a, b)| a | b)
        .collect();
    let exclude_tet: Vec<u8> = keyframes[i0]
        .4
        .iter()
        .zip(keyframes[i1].4.iter())
        .map(|(a, b)| a | b)
        .collect();
    RestShapeUpdate {
        inv_rest2x2: CVec::from(&inv2[..]),
        inv_rest3x3: CVec::from(&inv3[..]),
        exclude_face: CVec::from(&exclude_face[..]),
        exclude_tet: CVec::from(&exclude_tet[..]),
    }
}

/// One precomputed material keyframe: `(time, face_params)`.
///
/// Each entry is a full face table assembled by `Scene::make_props_at` at that
/// keyframe's time, so it went through the same assembly as the build-time
/// table and carries the same derived quantities.
struct MaterialKeyframe {
    time: f64,
    face: Vec<FaceParam>,
    vertex: Vec<VertexParam>,
    edge: Vec<EdgeParam>,
    hinge: Vec<HingeParam>,
}

/// Interpolate a material keyframe schedule to `time` and pack every table for
/// `update_material_params`. Clamps to the nearest end outside the schedule.
///
/// Every float field is blended, not a chosen subset. A field no animated key
/// can reach holds the same value in both keyframes, so blending it is a
/// no-op, and that is worth more than the saved arithmetic: a per-field
/// allowlist has to be extended whenever a key becomes animatable, and
/// forgetting one leaves that parameter frozen at its first keyframe with
/// nothing to show it. `model` and the boolean flags are structural and come
/// from the earlier keyframe.
///
/// Blending the DERIVED tables is exact rather than an approximation: a hinge,
/// edge or vertex value is an area- or length-weighted average of face values,
/// which is linear, so averaging blended faces and blending averaged hinges
/// agree. Only the face table's own Lame conversion is nonlinear, and there
/// the keyframes are what the artist set (see `interp_rest_shape`, which makes
/// the same choice for assembled inverse-rest matrices).
fn interp_material_params(
    keyframes: &[MaterialKeyframe],
    time: f64,
) -> (Vec<FaceParam>, Vec<VertexParam>, Vec<EdgeParam>, Vec<HingeParam>) {
    let n = keyframes.len();
    let (lo, hi, alpha) = if n == 1 || time <= keyframes[0].time {
        (0, 0, 0.0f32)
    } else if time >= keyframes[n - 1].time {
        (n - 1, n - 1, 0.0f32)
    } else {
        let hi = keyframes.partition_point(|k| k.time < time);
        let lo = hi - 1;
        let (t0, t1) = (keyframes[lo].time, keyframes[hi].time);
        let a = if t1 > t0 {
            ((time - t0) / (t1 - t0)) as f32
        } else {
            0.0
        };
        (lo, hi, a)
    };
    let f = |a: f32, b: f32| a + (b - a) * alpha;
    let (ka, kb) = (&keyframes[lo], &keyframes[hi]);

    let face = ka
        .face
        .iter()
        .zip(kb.face.iter())
        .map(|(a, b)| FaceParam {
            model: a.model,
            mu: f(a.mu, b.mu),
            lambda: f(a.lambda, b.lambda),
            friction: f(a.friction, b.friction),
            ghat: f(a.ghat, b.ghat),
            offset: f(a.offset, b.offset),
            bend: f(a.bend, b.bend),
            strainlimit: f(a.strainlimit, b.strainlimit),
            shrink_x: f(a.shrink_x, b.shrink_x),
            shrink_y: f(a.shrink_y, b.shrink_y),
            pressure: f(a.pressure, b.pressure),
            plasticity: f(a.plasticity, b.plasticity),
            plasticity_threshold: f(a.plasticity_threshold, b.plasticity_threshold),
            bend_plasticity: f(a.bend_plasticity, b.bend_plasticity),
            bend_plasticity_threshold: f(
                a.bend_plasticity_threshold,
                b.bend_plasticity_threshold,
            ),
            bend_rest_from_geometry: a.bend_rest_from_geometry,
            deform_damping: f(a.deform_damping, b.deform_damping),
            bend_damping: f(a.bend_damping, b.bend_damping),
            bend_warp: f(a.bend_warp, b.bend_warp),
            bend_weft: f(a.bend_weft, b.bend_weft),
        })
        .collect();

    let vertex = ka
        .vertex
        .iter()
        .zip(kb.vertex.iter())
        .map(|(a, b)| VertexParam {
            ghat: f(a.ghat, b.ghat),
            offset: f(a.offset, b.offset),
            friction: f(a.friction, b.friction),
        })
        .collect();

    let edge = ka
        .edge
        .iter()
        .zip(kb.edge.iter())
        .map(|(a, b)| EdgeParam {
            stiffness: f(a.stiffness, b.stiffness),
            bend: f(a.bend, b.bend),
            ghat: f(a.ghat, b.ghat),
            offset: f(a.offset, b.offset),
            friction: f(a.friction, b.friction),
            strainlimit: f(a.strainlimit, b.strainlimit),
            plasticity: f(a.plasticity, b.plasticity),
            plasticity_threshold: f(a.plasticity_threshold, b.plasticity_threshold),
            bend_rest_from_geometry: a.bend_rest_from_geometry,
            deform_damping: f(a.deform_damping, b.deform_damping),
            bend_damping: f(a.bend_damping, b.bend_damping),
        })
        .collect();

    let hinge = ka
        .hinge
        .iter()
        .zip(kb.hinge.iter())
        .map(|(a, b)| HingeParam {
            bend: f(a.bend, b.bend),
            ghat: f(a.ghat, b.ghat),
            offset: f(a.offset, b.offset),
            plasticity: f(a.plasticity, b.plasticity),
            plasticity_threshold: f(a.plasticity_threshold, b.plasticity_threshold),
            bend_damping: f(a.bend_damping, b.bend_damping),
            bend_warp: f(a.bend_warp, b.bend_warp),
            bend_weft: f(a.bend_weft, b.bend_weft),
        })
        .collect();

    (face, vertex, edge, hinge)
}

/// Supplied only by verified runtime model/selection overlay, never by a sidecar.
#[derive(Clone)]
pub(crate) struct TwistCheckpointContext {
    pub scene:String, pub joints:Vec<String>, pub sealed_models:Vec<String>,
    pub selected:Vec<String>, pub mask:Vec<bool>,
    pub mode:crate::checkpoint_history::TwistModeMetadata, pub explicit_migration:bool,
}
impl TwistCheckpointContext {
    fn principal(&self)->Vec<f64> {
        if self.mask.len()!=self.joints.len() || self.joints.iter().zip(&self.mask).filter_map(|(id,selected)|selected.then_some(id.clone())).collect::<Vec<_>>()!=self.selected {
            panic!("CHECKPOINT_TWIST_INVALID: trusted mask/order mismatch");
        }
        crate::driver::sample_committed_twist(&self.mask).unwrap_or_else(|e|panic!("checkpoint twist pose: {}",e.detail)).into_iter().map(|r|r[0]).collect()
    }
}
#[derive(Serialize, Deserialize)]
pub struct Backend {
    pub mesh: MeshSet,
    pub state: State,
    #[serde(skip)]
    pub(crate) connected_collision:Option<crate::connected_collision_native::Marker>,
    #[serde(skip)]
    pub(crate) resume_sand:Option<Vec<f32>>,
    #[serde(skip)]
    pub(crate) resume_history:Option<crate::checkpoint_history::History>,
    #[serde(skip)]
    pub(crate) resume_twist_payload:Option<crate::checkpoint_history::HistoryPayload>,
    #[serde(skip)]
    pub(crate) twist_checkpoint_context:Option<TwistCheckpointContext>,
    #[serde(skip)]
    pub(crate) material_spin_overlay:Option<crate::twist_overlay::TwistOverlay>,
    #[serde(skip)]
    pub(crate) fem_material_spin_overlay:Option<crate::twist_overlay::TwistOverlay>,
    #[serde(skip)]
    pub(crate) sealed_initial:Option<Vec<f32>>,
    #[serde(skip)]
    checkpoint_adaptive:Option<crate::driver::adaptive_controller::Controller>,
    #[serde(skip)]
    checkpoint_adaptive_start:f64,
    #[serde(skip)]
    checkpoint_context:Option<(String,String,Vec<u8>)>,
    #[serde(skip)]
    checkpoint_filtered:usize,
    #[serde(skip)]
    checkpoint_accepted:u64,
}

#[derive(Serialize, Deserialize)]
pub struct MeshSet {
    pub mesh: Mesh,
    pub uv: Option<Vec<Matrix2x3<f32>>>,
    pub vertex: Matrix3xX<f32>,
    pub rest_vertex: Option<Matrix3xX<f32>>,
    /// Per-object bending reference rest shape: hinge rest angles for the
    /// objects marked in `bend_rest_vertex_mask` are computed from these
    /// positions instead of the initial `vertex`. Unmasked columns equal the
    /// initial vert, so it is a no-op for non-reference objects. `None` when
    /// no object uses a reference rest angle.
    #[serde(default)]
    pub bend_rest_vertex: Option<Matrix3xX<f32>>,
    /// Per-vertex mask (length = vertex count, or empty when
    /// `bend_rest_vertex` is `None`) marking vertices whose hinges take their
    /// rest angle from `bend_rest_vertex`.
    #[serde(default)]
    pub bend_rest_vertex_mask: Vec<bool>,
    /// Per-vertex mask (length = vertex count, or empty when the scene has no
    /// STATIC collider in the solved namespace) marking vertices that belong to
    /// a collider. Contact and intersection reporting skip a pair only when
    /// BOTH sides are marked, so a collider still meets every dynamic object.
    /// `serde(default)` so a checkpoint carrying no such mask loads as "no
    /// colliders", the same as a scene with no STATIC collider.
    #[serde(default)]
    pub collider_vertex_mask: Vec<u8>,
    /// Per-vertex source-object identity (length = vertex count, or empty when
    /// the session directory carries no `object_vert.bin`). Separates a self-
    /// intersection from an inter-object one; empty means the solver knows no
    /// object boundaries and can grant neither allowance. `serde(default)` so
    /// a checkpoint whose serialized form omits it deserializes as empty.
    #[serde(default)]
    pub object_vertex_index: Vec<u32>,
    /// Per-vertex source-group identity (length = vertex count, or empty when
    /// the session directory carries no `group_vert.bin`, which the frontend
    /// writes only for a scene that asks for the inter-group allowance).
    /// Empty puts every vertex in one group. `serde(default)` for the same
    /// reason as `object_vertex_index`.
    #[serde(default)]
    pub group_vertex_index: Vec<u32>,
    /// Per-vertex intersection allowances resolved from each vertex's object's
    /// material (`INTERSECT_ALLOW_SELF | INTERSECT_ALLOW_INTER_OBJECT |
    /// INTERSECT_ALLOW_INTER_GROUP`).
    /// Empty means every object is at the default, which allows nothing.
    #[serde(default)]
    pub intersect_policy: Vec<u8>,
    #[serde(default)]
    pub collision_filter: Vec<u32>,
    /// Allow Existing Intersections' vertex links, flat `u32` pairs in the
    /// combined namespace (the dynamic vertices, then the collision-mesh
    /// vertices after them), exactly as `start_link.bin` carries them.
    /// `builder::start_link_table` turns them into the per-vertex table the
    /// contact passes read. Empty means nothing is linked.
    #[serde(default)]
    pub start_link: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
pub struct State {
    pub curr_vertex: Matrix3xX<f32>,
    pub prev_vertex: Matrix3xX<f32>,
    pub dyn_index: Vec<u32>,
    pub dyn_offset: Vec<u32>,
    pub time: f64,
    pub prev_dt: f32,
    pub curr_frame: i32,
}

impl Backend {
    pub fn new(mesh: MeshSet) -> Self {
        let state = State {
            curr_vertex: mesh.vertex.clone(),
            prev_vertex: mesh.vertex.clone(),
            dyn_index: Vec::new(),
            dyn_offset: Vec::new(),
            time: 0.0,
            prev_dt: 1.0,
            curr_frame: -1,
        };
        let num_face = mesh.mesh.mesh.face.ncols();
        let num_tet = mesh.mesh.mesh.tet.ncols();
        info!(
            "#v = {}, #f = {}, #tet = {}",
            mesh.mesh.mesh.vertex_count, num_face, num_tet
        );
        Self { state, mesh, connected_collision:None,resume_sand:None,resume_history:None,resume_twist_payload:None,twist_checkpoint_context:None,material_spin_overlay:None,fem_material_spin_overlay:None,sealed_initial:None,checkpoint_adaptive:None,checkpoint_adaptive_start:0.,checkpoint_context:None,checkpoint_filtered:0,checkpoint_accepted:0 }
    }

    fn fetch_state(&mut self, dataset: &DataSet, param: &ParamSet) {
        // Per-frame readback only needs vertex.prev/vertex.curr (fetch())
        // plus the dyn arrays below. The inv_rest / rest-angle DtoH copies
        // are serialized solely by save_state, so they live there now to
        // keep this per-frame write path off that extra bandwidth.
        unsafe {
            fetch();
        }
        let prev_vertex = unsafe {
            Matrix3xX::from_column_slice(std::slice::from_raw_parts(
                dataset.vertex.prev.data as *const f32,
                3 * dataset.vertex.prev.size as usize,
            ))
        };
        let curr_vertex = unsafe {
            Matrix3xX::from_column_slice(std::slice::from_raw_parts(
                dataset.vertex.curr.data as *const f32,
                3 * dataset.vertex.curr.size as usize,
            ))
        };
        let prev_dt = param.prev_dt;
        let mut n_value = 0;
        let mut n_offset = 0;
        unsafe {
            fetch_dyn_counts(&mut n_value, &mut n_offset);
        }
        self.state.dyn_index.resize(n_value as usize, 0);
        self.state.dyn_offset.resize(n_offset as usize, 0);
        unsafe {
            fetch_dyn(
                self.state.dyn_index.as_mut_ptr(),
                std::ptr::null_mut(),
                self.state.dyn_offset.as_mut_ptr(),
            );
        }
        self.state.prev_vertex = prev_vertex;
        self.state.curr_vertex = curr_vertex;
        self.state.prev_dt = prev_dt;
    }

    pub fn load_state(frame: i32, dirpath: &str) -> Self {
        let path_state=format!("{dirpath}/{}",ppf_cts_formats::files::state_filename(frame));
        let bytes=std::fs::read(&path_state).expect("checkpoint State read");
        let decoded=crate::checkpoint_state_marker::decode(&bytes).unwrap_or_else(|e|panic!("{e}"));
        if decoded.sand.is_some(){panic!("CHECKPOINT_SAND_INVALID: use validated resume path for grain checkpoint");}
        Self::load_state_payload(dirpath,&decoded.payload)
    }

    /// Deserialize the exact already-decoded State whose provenance main validated.
    /// Mesh remains a separate immutable build-time checkpoint artifact.
    pub fn load_state_payload(dirpath: &str, payload: &[u8]) -> Self {
        let path_mesh=format!("{dirpath}/{}",ppf_cts_formats::files::MESHSET);
        let mesh=super::read(&super::read_gz(&path_mesh));
        let state=super::read(payload);
        Self { state, mesh, connected_collision:None,resume_sand:None,resume_history:None,resume_twist_payload:None,twist_checkpoint_context:None,material_spin_overlay:None,fem_material_spin_overlay:None,sealed_initial:None,checkpoint_adaptive:None,checkpoint_adaptive_start:0.,checkpoint_context:None,checkpoint_filtered:0,checkpoint_accepted:0 }
    }

    /// Write `vert_<n>.bin`, `time_per_frame.out`, and
    /// `frame_to_time.out` for one finalized frame and run the optional
    /// auto-save side-effects. Extracted from the per-iteration loop in
    /// `run` so the rest pose (frame 0) and post-advance frames share
    /// the same write path.
    ///
    /// The frame is emitted at the EXACT frame time `t_frame` (= `frame /
    /// fps`), not at the variable, TOI-limited post-step time the solver
    /// happened to land on. The step that crossed this frame boundary ran
    /// from `t_prev` to `t_curr`, bracketing it, so the written vertex
    /// positions are linearly interpolated between `prev_vertex` and
    /// `curr_vertex` to `t_frame`. This keeps each output frame on the
    /// scene's framerate base instead of drifting by up to one substep,
    /// which otherwise shows up as a sawtooth in relative frame time. For
    /// the rest pose (frame 0) the caller passes `t_prev == t_curr`, which
    /// collapses the interpolation to `curr_vertex`.
    ///
    /// A fix-pinned vertex is the exception, and is placed exactly rather than
    /// interpolated: see `pin_at_frame` below.
    fn write_frame_outputs(
        &mut self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        dataset: &DataSet,
        param: &ParamSet,
        scene: &Scene,
        last_time: &mut Instant,
        frame: i32,
        t_frame: f64,
        t_prev: f64,
        t_curr: f64,
        statistics: &mut Option<crate::statistics::TimelineStatistics>,
        pose_accepted_steps: u64,
        publisher: &mut Option<crate::frame_output::Publisher>,
        publish_progress: bool,
    ) {
        if let Some(publisher) = publisher.as_mut() {
            self.snapshot_frame_outputs(program_args,sim_args,dataset,param,scene,last_time,frame,t_frame,t_prev,t_curr,statistics,pose_accepted_steps,publisher,publish_progress);
            return;
        }
        // Name: Time Per Video Frame
        // Format: list[(frame, ms)]
        // Description:
        // Wall-clock time in milliseconds elapsed between producing
        // the previous output video frame and this one. Because each
        // simulation step can advance less than one frame's worth of
        // time, this aggregates however many solver steps were needed
        // to cross the frame boundary.
        /*== push "time_per_frame" ==*/
        let mut time_per_frame = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}/data/time_per_frame.out", program_args.output).as_str())
            .unwrap();
        // Name: Video Frame to Simulation Time Map
        // Format: list[(frame, seconds)]
        // Description:
        // Pairs of (video frame index, simulation time in seconds)
        // emitted once per output frame. Lets you look up which
        // simulation time each output frame corresponds to, which is
        // useful when the playback rate varies over time.
        /*== push "frame_to_time" ==*/
        let mut frame_to_time = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}/data/frame_to_time.out", program_args.output).as_str())
            .unwrap();
        let curr_time = Instant::now();
        let elapsed = curr_time - *last_time;
        self.fetch_state(dataset, param);
        // Stamp the frame this fetched state belongs to before any
        // save_state below. The run loop only advances self.state.curr_frame
        // to the batch's final frame *after* write_frame_outputs returns, so
        // without this an auto-save or checkpoint here would name its file
        // state_<frame - 1> (the previous frame), surfacing as an off-by-one
        // in the Resume-From dialog. The loop's frame range is built before
        // this call, so updating curr_frame here does not affect iteration.
        self.state.curr_frame = frame;
        writeln!(time_per_frame, "{} {}", frame, elapsed.as_millis()).unwrap();
        writeln!(frame_to_time, "{} {}", frame, t_frame).unwrap();
        let path = format!(
            "{}/{}.tmp",
            program_args.output,
            ppf_cts_formats::files::vert_filename(frame)
        );
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.clone())
            .unwrap();
        let mut writer = std::io::BufWriter::with_capacity(super::IO_BUF_CAPACITY, file);
        // A faceless point cloud (e.g. a SAND grain set) derives no surface
        // vertices from elements, so vertex_count is 0. Fall back to the full
        // cloud width so the loose verts still reach vert_<i>.bin.
        let surface_vert_count = self
            .mesh
            .mesh
            .mesh
            .vertex_count
            .max(self.mesh.vertex.ncols());
        // Undo the world-scaling applied to all input geometry on ingest
        // (scene.rs), so the written positions are at the user's authored scale.
        // A 1.0 factor (the default) makes this a no-op.
        let inv_world = if sim_args.world_scaling > 0.0 {
            1.0 / sim_args.world_scaling
        } else {
            1.0
        };
        // Linear-interpolation weight that lands the written positions on the
        // exact frame time. `prev_vertex` is the pose at `t_prev` (step start),
        // `curr_vertex` the pose at `t_curr` (step end, which overshoots the
        // boundary); alpha places the output at `t_frame` in between. A zero
        // span (frame 0 rest pose, or a degenerate step) writes `curr_vertex`
        // verbatim. This is output-only: `self.state.*_vertex` stay as the true
        // GPU readback so `save_state` checkpoints the real solver state.
        let alpha = frame_interp_alpha(t_frame, t_prev, t_curr);

        // A fix-pinned vertex rides a prescribed path, so where it belongs at
        // t_frame is known outright. Interpolating its absolute position
        // between the two steps bracketing the frame instead draws a chord
        // across that path and cuts the corner, by up to dt/4 times the path's
        // curvature. The step boundaries fall wherever the CCD line search
        // leaves them, so the size of that cut varies frame to frame, and on a
        // driven collider it reads as a jitter over an animation that should be
        // perfectly smooth. It is most visible against a camera that shares the
        // collider's motion: the camera samples the exact keyframe path while
        // the collider is displayed from the chord, and the mismatch between
        // them is exactly the jiggle.
        //
        // So place the pin at its exact pose for t_frame and interpolate only
        // the residual: how far the solver's readback sits off that path. For
        // an exact-Dirichlet fix pin the residual is zero to round-off (its DOF
        // are eliminated, so the solver holds it dead on the path), which is
        // why the collider now tracks its keyframe exactly. For the one fix pin
        // still held by a barrier (a PDRD anchor) the residual is small and
        // smooth, so interpolating it is genuinely second-order. For a pin that
        // never moves the residual identity collapses this back to the plain
        // lerp.
        //
        // Only vertices pinned across the whole step qualify; one whose pin
        // starts or is released mid-step is left to the lerp for that frame.
        let mut pin_at_frame: HashMap<u32, (Vec3f, Vec3f, Vec3f)> = {
            let at_prev: HashMap<u32, Vec3f> =
                scene.fix_pin_positions(t_prev).into_iter().collect();
            let at_curr: HashMap<u32, Vec3f> =
                scene.fix_pin_positions(t_curr).into_iter().collect();
            scene
                .fix_pin_positions(t_frame)
                .into_iter()
                .filter_map(|(i, at_f)| match (at_prev.get(&i), at_curr.get(&i)) {
                    (Some(&p), Some(&c)) => Some((i, (at_f, p, c))),
                    _ => None,
                })
                .collect()
        };

        // Freeze segment evidence while still on the solver thread. Async
        // publisher receives encoded bytes only, never mutable control state.
        let fem_control_output=if scene.joint_input.as_ref().is_some_and(|i|i.contains(crate::fem_control_events::FILE)) {
            crate::driver::fem_control_output(t_frame,t_prev,t_curr).expect("FEM accepted segment output")
        }else{None};
        if let Some(controls)=&fem_control_output {for &(index,(f,p,c)) in &controls.pins {pin_at_frame.insert(index,(Vec3f::from(f.map(|x|x as f32)),Vec3f::from(p.map(|x|x as f32)),Vec3f::from(c.map(|x|x as f32))));}}

        // Write visible vertices in chunks while retaining the exact full pose
        // for solver-side object statistics. Interior SOLID vertices are not
        // written to PC2, but volume analysis needs them.
        const CHUNK_SIZE: usize = 4096;
        let mut chunk_buf: Vec<f32> = Vec::with_capacity(CHUNK_SIZE);
        let mut output_positions = Vec::with_capacity(self.state.curr_vertex.ncols());
        for v in 0..self.state.curr_vertex.ncols() {
            let p = self.state.prev_vertex.column(v);
            let c = self.state.curr_vertex.column(v);
            let mut output = [0.0; 3];
            let mut analysis_output = [0.0_f64; 3];
            match pin_at_frame.get(&(v as u32)) {
                Some((at_f, at_p, at_c)) => {
                    for k in 0..3 {
                        // INTERPOLATE THE RESIDUAL, not the absolute position:
                        // the exact scripted pose is added back at the end, so
                        // the interpolation runs on a small offset rather than
                        // on two large nearby coordinates whose difference
                        // would lose its leading digits.
                        let rp = f32::from(p[k] - at_p[k]);
                        let rc = f32::from(c[k] - at_c[k]);
                        let exact = f32::from(at_f[k]);
                        output[k] = (exact + rp + alpha * (rc - rp)) * inv_world;
                        let rp64 = f64::from(p[k] - at_p[k]);
                        let rc64 = f64::from(c[k] - at_c[k]);
                        let exact64 = f64::from(at_f[k]);
                        analysis_output[k] =
                            (exact64 + rp64 + alpha as f64 * (rc64 - rp64)) * inv_world as f64;
                    }
                }
                None => {
                    for k in 0..3 {
                        let pf = f32::from(p[k]);
                        let cf = f32::from(c[k]);
                        output[k] = (pf + alpha * (cf - pf)) * inv_world;
                        let pf64 = f64::from(p[k]);
                        let cf64 = f64::from(c[k]);
                        analysis_output[k] =
                            (pf64 + alpha as f64 * (cf64 - pf64)) * inv_world as f64;
                    }
                }
            }
            output_positions.push(analysis_output);
            if v < surface_vert_count {
                chunk_buf.extend_from_slice(&output);
            }
            if chunk_buf.len() >= CHUNK_SIZE {
                let buff = unsafe {
                    std::slice::from_raw_parts(
                        chunk_buf.as_ptr() as *const u8,
                        chunk_buf.len() * std::mem::size_of::<f32>(),
                    )
                };
                writer.write_all(buff).unwrap();
                chunk_buf.clear();
            }
        }
        if !chunk_buf.is_empty() {
            let buff = unsafe {
                std::slice::from_raw_parts(
                    chunk_buf.as_ptr() as *const u8,
                    chunk_buf.len() * std::mem::size_of::<f32>(),
                )
            };
            writer.write_all(buff).unwrap();
        }
        writer.flush().unwrap();
        if let Some(statistics) = statistics {
            statistics.analyze_and_write(
                frame as u32,
                t_frame,
                &output_positions,
                &self.mesh.mesh.mesh,
                dataset,
                inv_world as f64,
            );
        }
        // Build the final path explicitly rather than stripping ".tmp" with
        // String::replace, which would also rewrite any ".tmp" inside the
        // output directory path. This mirrors the temp-path construction above
        // with the same {output}/vert_{frame} stem.
        let final_path = format!(
            "{}/{}",
            program_args.output,
            ppf_cts_formats::files::vert_filename(frame)
        );
        std::fs::rename(&path, &final_path).unwrap();
        // Insert after the atomic rename of vert_<frame>.bin and before remove_old_files.
        // `pose_accepted_steps` is an extra write_frame_outputs argument: 0 at frame 0,
        // joint_accepted after successful advance at the ordinary saved-frame call.
        #[cfg(neutral_driver)]
        {
        // A material-only carrier has no rigid pose to measure or publish.
        // Its independent FEM block below still runs for every saved frame.
        if let Some(input)=scene.joint_input.as_ref().filter(|input| !input.records().bodies.is_empty()) {
            let owned=input.records();
            let vertices=std::fs::read(&final_path).expect("read exact published pose vertex frame");
            assert_eq!(vertices.len()%12,0,"pose frame is not f32 xyz");
            let positions:Vec<f32>=vertices.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
            let native=crate::driver::pose_output::measure_frame_positions(&positions,sim_args.world_scaling)
                .expect("native read-only frame pose measurement");
            let bindings:Vec<_>=owned.bodies.iter().map(|(uid,index)|crate::pose_output::Binding {
                body_uuid:uid.clone(),runtime_body_index:*index }).collect();
            let pose=crate::pose_output::PoseFrame::from_geometry_measurements(owned.scene_uuid.clone(),owned.scene_inputs_sha256.clone(),
                &vertices,u32::try_from(frame).expect("negative pose frame"),t_frame,t_prev,t_curr,
                pose_accepted_steps,&bindings,&native,sim_args.world_scaling).expect("accepted pose sidecar validation");
            crate::pose_output::publish(std::path::Path::new(&program_args.output),&pose,&vertices)
                .expect("accepted frame pose publication");
        }
        }
// Add `mod soft_pose_output;` beside pose_output in main.rs.
// Insert after atomic vert_<frame>.bin rename, before remove_old_files,
// alongside (NOT inside) existing PDRD pose block. Published bytes are meters;
// do not reapply world_scaling. `prepare` validates retained immutable inputs.
#[cfg(neutral_driver)]
if let Some(input)=scene.joint_input.as_ref() {
    if let Some(soft)=crate::soft_joint::prepare(input).expect("FEM retained input identity") {
        let bindings=crate::soft_pose_output::bindings_from_owned(&soft).expect("FEM pose binding");
        if !bindings.is_empty() {
            let vertices=std::fs::read(&final_path).expect("exact published FEM vertex generation");
            let mut pose=crate::soft_pose_output::PoseFrame::from_vertices(
                soft.scene_uuid.clone(),soft.scene_inputs_sha256.clone(),soft.payload_sha256.clone(),
                &vertices,u32::try_from(frame).expect("negative FEM pose frame"),t_frame,t_prev,t_curr,
                pose_accepted_steps,&bindings).expect("accepted FEM material frame validation");
            if let Some(controls)=&fem_control_output {pose.version=2;pose.constrained_root_state_authority="native_fem_control_segments_v2".into();pose.control_program_sha256=Some(controls.program_sha256.clone());pose.control_segments=Some(controls.rows.clone());}
            crate::soft_pose_output::publish(std::path::Path::new(&program_args.output),&pose,&vertices)
                .expect("accepted FEM material pose publication");
        }
    }
}

        super::remove_old_files(
            &program_args.output,
            ppf_cts_formats::files::VERT_PREFIX,
            ppf_cts_formats::files::VERT_SUFFIX,
            sim_args.keep_verts,
            frame,
        );
        // Display pins: the scripted positions, at t_frame, of the Blender
        // vertices exact SOLID pins hold (see `Scene::display_pin_positions`).
        // The file is one byte per block, 1 while the block is still pinned
        // and 0 once it is released, followed by every position as three
        // float32s at the user's authored scale, blocks in order. Its layout
        // is fixed for the scene, so a reader checks the size against the
        // counts the scene was exported with.
        if scene.has_display_pins() {
            let (active, positions) = scene.display_pin_positions(t_frame);
            let mut bytes: Vec<u8> = Vec::with_capacity(active.len() + 12 * positions.len());
            bytes.extend(active.iter().map(|&a| u8::from(a)));
            for p in positions.iter() {
                for k in 0..3 {
                    bytes.extend_from_slice(&(f32::from(p[k]) * inv_world).to_le_bytes());
                }
            }
            let final_path = format!(
                "{}/{}",
                program_args.output,
                ppf_cts_formats::files::display_pin_filename(frame)
            );
            let tmp_path = format!("{final_path}.tmp");
            std::fs::write(&tmp_path, &bytes).unwrap();
            std::fs::rename(&tmp_path, &final_path).unwrap();
            super::remove_old_files(
                &program_args.output,
                ppf_cts_formats::files::DISPLAY_PIN_PREFIX,
                ppf_cts_formats::files::DISPLAY_PIN_SUFFIX,
                sim_args.keep_verts,
                frame,
            );
        }
        // Save a resumable state when this frame hits the auto-save cadence
        // or appears in the explicit checkpoint list. Both use the same
        // `frame` convention so a checkpoint saves the exact frame the
        // artist requested, and a frame matched by both saves only once.
        //
        // Auto-save counts the 1-based output frame: `frame` is solver 0-based
        // with frame 0 the rest pose, and the UI displays it as frame+1, so an
        // interval N must land on the displayed frames N, 2N, 3N (solver frames
        // N-1, 2N-1, ...). Using `(frame + 1) % N == 0` makes the cadence match
        // what the user sees and the Blender->solver checkpoint conversion
        // (display N -> solver N-1).
        let is_auto_save = sim_args.auto_save > 0 && (frame + 1) % sim_args.auto_save == 0;
        let is_checkpoint = sim_args
            .checkpoints
            .split(',')
            .filter_map(|t| t.trim().parse::<i32>().ok())
            .any(|c| c == frame);
        if frame > 0 && (is_auto_save || is_checkpoint) {
            info!("saving state (auto_save={is_auto_save}, checkpoint={is_checkpoint})...");
            self.save_state(program_args, sim_args, dataset,param);
        }
        *last_time = Instant::now();
    }

    fn snapshot_frame_outputs(
        &mut self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        dataset: &DataSet,
        param: &ParamSet,
        scene: &Scene,
        last_time: &mut Instant,
        frame: i32,
        t_frame: f64,
        t_prev: f64,
        t_curr: f64,
        statistics: &mut Option<crate::statistics::TimelineStatistics>,
        pose_accepted_steps: u64,
        publisher: &mut crate::frame_output::Publisher,
        publish_progress: bool,
    ) {
        // Name: Time Per Video Frame
        // Format: list[(frame, ms)]
        // Description:
        // Wall-clock time in milliseconds elapsed between producing
        // the previous output video frame and this one. Because each
        // simulation step can advance less than one frame's worth of
        // time, this aggregates however many solver steps were needed
        // to cross the frame boundary.
        /*== push "time_per_frame" ==*/
        let mut time_per_frame = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}/data/time_per_frame.out", program_args.output).as_str())
            .unwrap();
        // Name: Video Frame to Simulation Time Map
        // Format: list[(frame, seconds)]
        // Description:
        // Pairs of (video frame index, simulation time in seconds)
        // emitted once per output frame. Lets you look up which
        // simulation time each output frame corresponds to, which is
        // useful when the playback rate varies over time.
        /*== push "frame_to_time" ==*/
        let mut frame_to_time = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}/data/frame_to_time.out", program_args.output).as_str())
            .unwrap();
        let curr_time = Instant::now();
        let elapsed = curr_time - *last_time;
        self.fetch_state(dataset, param);
        // Stamp the frame this fetched state belongs to before any
        // save_state below. The run loop only advances self.state.curr_frame
        // to the batch's final frame *after* write_frame_outputs returns, so
        // without this an auto-save or checkpoint here would name its file
        // state_<frame - 1> (the previous frame), surfacing as an off-by-one
        // in the Resume-From dialog. The loop's frame range is built before
        // this call, so updating curr_frame here does not affect iteration.
        self.state.curr_frame = frame;
        writeln!(time_per_frame, "{} {}", frame, elapsed.as_millis()).unwrap();
        writeln!(frame_to_time, "{} {}", frame, t_frame).unwrap();
        let mut writer = Vec::<u8>::new();
        // A faceless point cloud (e.g. a SAND grain set) derives no surface
        // vertices from elements, so vertex_count is 0. Fall back to the full
        // cloud width so the loose verts still reach vert_<i>.bin.
        let surface_vert_count = self
            .mesh
            .mesh
            .mesh
            .vertex_count
            .max(self.mesh.vertex.ncols());
        // Undo the world-scaling applied to all input geometry on ingest
        // (scene.rs), so the written positions are at the user's authored scale.
        // A 1.0 factor (the default) makes this a no-op.
        let inv_world = if sim_args.world_scaling > 0.0 {
            1.0 / sim_args.world_scaling
        } else {
            1.0
        };
        // Linear-interpolation weight that lands the written positions on the
        // exact frame time. `prev_vertex` is the pose at `t_prev` (step start),
        // `curr_vertex` the pose at `t_curr` (step end, which overshoots the
        // boundary); alpha places the output at `t_frame` in between. A zero
        // span (frame 0 rest pose, or a degenerate step) writes `curr_vertex`
        // verbatim. This is output-only: `self.state.*_vertex` stay as the true
        // GPU readback so `save_state` checkpoints the real solver state.
        let alpha = frame_interp_alpha(t_frame, t_prev, t_curr);

        // A fix-pinned vertex rides a prescribed path, so where it belongs at
        // t_frame is known outright. Interpolating its absolute position
        // between the two steps bracketing the frame instead draws a chord
        // across that path and cuts the corner, by up to dt/4 times the path's
        // curvature. The step boundaries fall wherever the CCD line search
        // leaves them, so the size of that cut varies frame to frame, and on a
        // driven collider it reads as a jitter over an animation that should be
        // perfectly smooth. It is most visible against a camera that shares the
        // collider's motion: the camera samples the exact keyframe path while
        // the collider is displayed from the chord, and the mismatch between
        // them is exactly the jiggle.
        //
        // So place the pin at its exact pose for t_frame and interpolate only
        // the residual: how far the solver's readback sits off that path. For
        // an exact-Dirichlet fix pin the residual is zero to round-off (its DOF
        // are eliminated, so the solver holds it dead on the path), which is
        // why the collider now tracks its keyframe exactly. For the one fix pin
        // still held by a barrier (a PDRD anchor) the residual is small and
        // smooth, so interpolating it is genuinely second-order. For a pin that
        // never moves the residual identity collapses this back to the plain
        // lerp.
        //
        // Only vertices pinned across the whole step qualify; one whose pin
        // starts or is released mid-step is left to the lerp for that frame.
        let mut pin_at_frame: HashMap<u32, (Vec3f, Vec3f, Vec3f)> = {
            let at_prev: HashMap<u32, Vec3f> =
                scene.fix_pin_positions(t_prev).into_iter().collect();
            let at_curr: HashMap<u32, Vec3f> =
                scene.fix_pin_positions(t_curr).into_iter().collect();
            scene
                .fix_pin_positions(t_frame)
                .into_iter()
                .filter_map(|(i, at_f)| match (at_prev.get(&i), at_curr.get(&i)) {
                    (Some(&p), Some(&c)) => Some((i, (at_f, p, c))),
                    _ => None,
                })
                .collect()
        };

        // Freeze segment evidence while still on the solver thread. Async
        // publisher receives encoded bytes only, never mutable control state.
        let fem_control_output=if scene.joint_input.as_ref().is_some_and(|i|i.contains(crate::fem_control_events::FILE)) {
            crate::driver::fem_control_output(t_frame,t_prev,t_curr).expect("FEM accepted segment output")
        }else{None};
        if let Some(controls)=&fem_control_output {for &(index,(f,p,c)) in &controls.pins {pin_at_frame.insert(index,(Vec3f::from(f.map(|x|x as f32)),Vec3f::from(p.map(|x|x as f32)),Vec3f::from(c.map(|x|x as f32))));}}

        // Write visible vertices in chunks while retaining the exact full pose
        // for solver-side object statistics. Interior SOLID vertices are not
        // written to PC2, but volume analysis needs them.
        const CHUNK_SIZE: usize = 4096;
        let mut chunk_buf: Vec<f32> = Vec::with_capacity(CHUNK_SIZE);
        let mut output_positions = Vec::with_capacity(self.state.curr_vertex.ncols());
        for v in 0..self.state.curr_vertex.ncols() {
            let p = self.state.prev_vertex.column(v);
            let c = self.state.curr_vertex.column(v);
            let mut output = [0.0; 3];
            let mut analysis_output = [0.0_f64; 3];
            match pin_at_frame.get(&(v as u32)) {
                Some((at_f, at_p, at_c)) => {
                    for k in 0..3 {
                        // INTERPOLATE THE RESIDUAL, not the absolute position:
                        // the exact scripted pose is added back at the end, so
                        // the interpolation runs on a small offset rather than
                        // on two large nearby coordinates whose difference
                        // would lose its leading digits.
                        let rp = f32::from(p[k] - at_p[k]);
                        let rc = f32::from(c[k] - at_c[k]);
                        let exact = f32::from(at_f[k]);
                        output[k] = (exact + rp + alpha * (rc - rp)) * inv_world;
                        let rp64 = f64::from(p[k] - at_p[k]);
                        let rc64 = f64::from(c[k] - at_c[k]);
                        let exact64 = f64::from(at_f[k]);
                        analysis_output[k] =
                            (exact64 + rp64 + alpha as f64 * (rc64 - rp64)) * inv_world as f64;
                    }
                }
                None => {
                    for k in 0..3 {
                        let pf = f32::from(p[k]);
                        let cf = f32::from(c[k]);
                        output[k] = (pf + alpha * (cf - pf)) * inv_world;
                        let pf64 = f64::from(p[k]);
                        let cf64 = f64::from(c[k]);
                        analysis_output[k] =
                            (pf64 + alpha as f64 * (cf64 - pf64)) * inv_world as f64;
                    }
                }
            }
            output_positions.push(analysis_output);
            if v < surface_vert_count {
                chunk_buf.extend_from_slice(&output);
            }
            if chunk_buf.len() >= CHUNK_SIZE {
                let buff = unsafe {
                    std::slice::from_raw_parts(
                        chunk_buf.as_ptr() as *const u8,
                        chunk_buf.len() * std::mem::size_of::<f32>(),
                    )
                };
                writer.write_all(buff).unwrap();
                chunk_buf.clear();
            }
        }
        if !chunk_buf.is_empty() {
            let buff = unsafe {
                std::slice::from_raw_parts(
                    chunk_buf.as_ptr() as *const u8,
                    chunk_buf.len() * std::mem::size_of::<f32>(),
                )
            };
            writer.write_all(buff).unwrap();
        }
        writer.flush().unwrap();
        let statistics_bytes = if let Some(statistics) = statistics {
            Some(statistics.analyze_and_encode(
                frame as u32,
                t_frame,
                &output_positions,
                &self.mesh.mesh.mesh,
                dataset,
                inv_world as f64,
            ))
        } else { None };
        let vertices = writer;
        #[allow(unused_mut)]
        let mut rigid_pose = None;
        #[allow(unused_mut)]
        let mut fem_pose = None;
        // Insert after the atomic rename of vert_<frame>.bin and before remove_old_files.
        // `pose_accepted_steps` is an extra write_frame_outputs argument: 0 at frame 0,
        // joint_accepted after successful advance at the ordinary saved-frame call.
        #[cfg(neutral_driver)]
        {
        // A material-only carrier has no rigid pose to measure or publish.
        // Its independent FEM block below still runs for every saved frame.
        if let Some(input)=scene.joint_input.as_ref().filter(|input| !input.records().bodies.is_empty()) {
            let owned=input.records();
            assert_eq!(vertices.len()%12,0,"pose frame is not f32 xyz");
            let positions:Vec<f32>=vertices.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
            let native=crate::driver::pose_output::measure_frame_positions(&positions,sim_args.world_scaling)
                .expect("native read-only frame pose measurement");
            let bindings:Vec<_>=owned.bodies.iter().map(|(uid,index)|crate::pose_output::Binding {
                body_uuid:uid.clone(),runtime_body_index:*index }).collect();
            let pose=crate::pose_output::PoseFrame::from_geometry_measurements(owned.scene_uuid.clone(),owned.scene_inputs_sha256.clone(),
                &vertices,u32::try_from(frame).expect("negative pose frame"),t_frame,t_prev,t_curr,
                pose_accepted_steps,&bindings,&native,sim_args.world_scaling).expect("accepted pose sidecar validation");
            rigid_pose=Some(crate::pose_output::encode(&pose,&vertices).expect("encode immutable rigid pose"));
        }
        }
// Add `mod soft_pose_output;` beside pose_output in main.rs.
// Insert after atomic vert_<frame>.bin rename, before remove_old_files,
// alongside (NOT inside) existing PDRD pose block. Published bytes are meters;
// do not reapply world_scaling. `prepare` validates retained immutable inputs.
#[cfg(neutral_driver)]
if let Some(input)=scene.joint_input.as_ref() {
    if let Some(soft)=crate::soft_joint::prepare(input).expect("FEM retained input identity") {
        let bindings=crate::soft_pose_output::bindings_from_owned(&soft).expect("FEM pose binding");
        if !bindings.is_empty() {
            let mut pose=crate::soft_pose_output::PoseFrame::from_vertices(
                soft.scene_uuid.clone(),soft.scene_inputs_sha256.clone(),soft.payload_sha256.clone(),
                &vertices,u32::try_from(frame).expect("negative FEM pose frame"),t_frame,t_prev,t_curr,
                pose_accepted_steps,&bindings).expect("accepted FEM material frame validation");
            if let Some(controls)=&fem_control_output {pose.version=2;pose.constrained_root_state_authority="native_fem_control_segments_v2".into();pose.control_program_sha256=Some(controls.program_sha256.clone());pose.control_segments=Some(controls.rows.clone());}
            fem_pose=Some(crate::soft_pose_output::encode(&pose,&vertices).expect("encode immutable FEM pose"));
        }
    }
}

        let mut display_pins = None;
        // Display pins: the scripted positions, at t_frame, of the Blender
        // vertices exact SOLID pins hold (see `Scene::display_pin_positions`).
        // The file is one byte per block, 1 while the block is still pinned
        // and 0 once it is released, followed by every position as three
        // float32s at the user's authored scale, blocks in order. Its layout
        // is fixed for the scene, so a reader checks the size against the
        // counts the scene was exported with.
        if scene.has_display_pins() {
            let (active, positions) = scene.display_pin_positions(t_frame);
            let mut bytes: Vec<u8> = Vec::with_capacity(active.len() + 12 * positions.len());
            bytes.extend(active.iter().map(|&a| u8::from(a)));
            for p in positions.iter() {
                for k in 0..3 {
                    bytes.extend_from_slice(&(f32::from(p[k]) * inv_world).to_le_bytes());
                }
            }
            display_pins = Some(bytes);
        }
        publisher.submit(crate::frame_output::FramePacket { frame, vertices, statistics:statistics_bytes, rigid_pose, fem_pose, display_pins, keep_verts:sim_args.keep_verts, progress:if publish_progress {Some(t_curr)} else {None} }).expect("publish immutable frame snapshot");
        // Save a resumable state when this frame hits the auto-save cadence
        // or appears in the explicit checkpoint list. Both use the same
        // `frame` convention so a checkpoint saves the exact frame the
        // artist requested, and a frame matched by both saves only once.
        //
        // Auto-save counts the 1-based output frame: `frame` is solver 0-based
        // with frame 0 the rest pose, and the UI displays it as frame+1, so an
        // interval N must land on the displayed frames N, 2N, 3N (solver frames
        // N-1, 2N-1, ...). Using `(frame + 1) % N == 0` makes the cadence match
        // what the user sees and the Blender->solver checkpoint conversion
        // (display N -> solver N-1).
        let is_auto_save = sim_args.auto_save > 0 && (frame + 1) % sim_args.auto_save == 0;
        let is_checkpoint = sim_args
            .checkpoints
            .split(',')
            .filter_map(|t| t.trim().parse::<i32>().ok())
            .any(|c| c == frame);
        if frame > 0 && (is_auto_save || is_checkpoint) {
            info!("saving state (auto_save={is_auto_save}, checkpoint={is_checkpoint})...");
            publisher.barrier().expect("drain frames before checkpoint");
            self.save_state(program_args, sim_args, dataset,param);
        }
        *last_time = Instant::now();
    }

    fn save_state(&mut self, program_args: &ProgramArgs, sim_args: &SimArgs, dataset: &DataSet,param:&ParamSet) {
        use ppf_cts_formats::files;
        // Save/Quit may happen after accepted steps that crossed no output frame.
        // Refresh actual device curr/prev/dynamic history, never a stale frame mirror.
        self.fetch_state(dataset,param);
        if self.state.time.to_bits()!=param.time.to_bits(){panic!("CHECKPOINT_CLOCK_INVALID: host/device accepted boundary differs");}
        let history=self.checkpoint_context.as_ref().map(|(scene_uuid,scene_inputs_sha256,param_bytes)|{
            let runtime=crate::driver::export_checkpoint(self.state.time).unwrap_or_else(|e|panic!("{}",e.detail));
            crate::checkpoint_history::History{kind:"ppf.bone_checkpoint_history".into(),version:1,
                frame:self.state.curr_frame,accepted_time:self.state.time,previous_dt:self.state.prev_dt,
                sealed_frames:sim_args.frames,fps:sim_args.fps,param_bytes:param_bytes.clone(),scene_uuid:scene_uuid.clone(),scene_inputs_sha256:scene_inputs_sha256.clone(),
                filtered_links:self.checkpoint_filtered,joint_accepted:self.checkpoint_accepted,
                adaptive:self.checkpoint_adaptive.clone().unwrap_or_else(||crate::driver::adaptive_controller::Controller::new(crate::builder::make_param(sim_args).dt)),adaptive_enabled:adaptive_cg_retry_enabled(),adaptive_start_time:self.checkpoint_adaptive_start,runtime}
        });
        let sand_mask=crate::checkpoint_sand::mask(dataset).unwrap_or_else(|e|panic!("{e}"));
        if sand_mask.is_some() && history.is_none(){panic!("CHECKPOINT_SAND_INVALID: accepted history context missing");}
        let transaction=history.map(|history|{
            let payload=if self.material_spin_overlay.is_some()||self.fem_material_spin_overlay.is_some() {
                let rigid=crate::driver::export_accepted_material_spin().unwrap_or_else(|e|panic!("checkpoint material spin export: {}",e.detail));
                let fem=crate::driver::export_accepted_fem_material_spin().unwrap_or_else(|e|panic!("checkpoint FEM material spin export: {}",e.detail));
                match (&self.material_spin_overlay,&rigid){(Some(ctx),Some(h))=>h.validate(&ctx.scene_uuid,&ctx.selected_joint_uuids,history.accepted_time).unwrap_or_else(|e|panic!("{e}")),(None,None)=>{},_=>panic!("CHECKPOINT_MATERIAL_SPIN_INVALID: rigid history/selection mismatch")}
                match (&self.fem_material_spin_overlay,&fem){(Some(ctx),Some(h))=>h.validate(&ctx.scene_uuid,&ctx.selected_joint_uuids,history.accepted_time).unwrap_or_else(|e|panic!("{e}")),(None,None)=>{},_=>panic!("CHECKPOINT_MATERIAL_SPIN_INVALID: FEM history/selection mismatch")}
                let twist=if let Some(ctx)=&self.twist_checkpoint_context {
                    let accepted=crate::driver::export_accepted_twist().unwrap_or_else(|e|panic!("checkpoint twist export: {}",e.detail)).expect("CHECKPOINT_TWIST_INVALID: missing accepted history");
                    match crate::checkpoint_history::HistoryPayload::continuous(history.clone(),accepted,ctx.mode.clone(),&ctx.scene,&ctx.joints,&ctx.principal(),&ctx.sealed_models,&ctx.selected).unwrap_or_else(|e|panic!("{e}")){crate::checkpoint_history::HistoryPayload::V2(v)=>Some(v),_=>unreachable!()}
                }else{if crate::driver::export_accepted_twist().unwrap_or_else(|e|panic!("checkpoint twist export: {}",e.detail)).is_some(){panic!("CHECKPOINT_TWIST_INVALID: missing trusted mode context");}None};
                let h=crate::checkpoint_history::HistoryV3{version:3,legacy:history,model:crate::material_spin_history::MODEL.into(),rigid_material:rigid,fem_material:fem,twist};
                crate::checkpoint_history::HistoryPayload::V3(h)
            } else if let Some(ctx)=&self.twist_checkpoint_context {
                let accepted=crate::driver::export_accepted_twist().unwrap_or_else(|e|panic!("checkpoint twist export: {}",e.detail)).expect("CHECKPOINT_TWIST_INVALID: missing accepted history");
                let principal=ctx.principal();
                crate::checkpoint_history::HistoryPayload::continuous(history,accepted,ctx.mode.clone(),&ctx.scene,&ctx.joints,&principal,&ctx.sealed_models,&ctx.selected).unwrap_or_else(|e|panic!("{e}"))
            } else {
                if crate::driver::export_accepted_twist().unwrap_or_else(|e|panic!("checkpoint twist export: {}",e.detail)).is_some() {panic!("CHECKPOINT_TWIST_INVALID: missing trusted mode context");}
                crate::checkpoint_history::HistoryPayload::V1(history)
            };
            let mut history_bytes=payload.encode().expect("checkpoint history serialize");
            // Grain states are encoded once after their spin block and layout
            // are ready; avoid compressing the same large State twice.
            let plastic=self.checkpoint_plastic_bytes(dataset);let output=Path::new(&program_args.output);
            let mut state_bytes=if sand_mask.is_none() {
                crate::checkpoint_state_marker::bind_plastic(&crate::checkpoint_state_marker::encode_state(&self.state,true).expect("checkpoint State serialize"),plastic.as_deref()).expect("plastic binding")
            } else { Vec::new() };
            if sand_mask.is_none() && output.join(files::state_filename(self.state.curr_frame)).exists(){
                let old=crate::checkpoint_history::read_versioned(output,self.state.curr_frame,true).unwrap_or_else(|e|panic!("{e}")).expect("required history");
                // Only explicit first migration at the identical accepted boundary may
                // augment a V1 history. State and plastic remain immutable.
                let migration=match (&payload,&old) {
                    (crate::checkpoint_history::HistoryPayload::V2(next),crate::checkpoint_history::HistoryPayload::V1(prev)) =>
                        next.legacy.semantic_eq(prev) && !next.twist.prior_turns_known &&
                        next.mode.migration_time==Some(next.legacy.accepted_time),
                    _ => false,
                };
                if !migration && !payload.semantic_eq(&old){panic!("CHECKPOINT_FRAME_IMMUTABLE: accepted histories differ");}
                let existing=std::fs::read(output.join(files::state_filename(self.state.curr_frame))).expect("existing State read");
                state_bytes=crate::checkpoint_state_marker::select_immutable_bytes(Some(&existing),&state_bytes)
                    .unwrap_or_else(|e|panic!("{e}")).into_owned();
                if migration {
                    let old_bytes=std::fs::read(output.join(crate::checkpoint_commit::history_name(self.state.curr_frame))).expect("existing history read");
                    let (_,old_payload)=crate::checkpoint_commit::decode_history(self.state.curr_frame,&old_bytes).unwrap_or_else(|e|panic!("{e}"));
                    if crate::checkpoint_history::decode_payload(old_payload).unwrap_or_else(|e|panic!("{e}"))!=old {panic!("CHECKPOINT_FRAME_IMMUTABLE: migration history changed");}
                    crate::checkpoint_commit::publish_migration_history(output,self.state.curr_frame,&state_bytes,&old_bytes,plastic.as_deref(),&history_bytes).unwrap_or_else(|e|panic!("{e}"));
                } else {
                    history_bytes=old.encode().expect("existing checkpoint serialize");
                    crate::checkpoint_commit::publish(output,self.state.curr_frame,&state_bytes,&history_bytes,plastic.as_deref()).unwrap_or_else(|e|panic!("{e}"));
                }
            }
            if let Some(mask)=sand_mask.as_ref().filter(|_|output.join(files::state_filename(self.state.curr_frame)).exists()) {
                // An existing checkpoint must not pass through dataset marker
                // invalidation. Compare its immutable accepted boundary first.
                crate::checkpoint_layout::validate(output).unwrap_or_else(|e|panic!("{e}"));
                let path=output.join(files::state_filename(self.state.curr_frame));
                let existing=std::fs::read(&path).expect("existing State read");
                let old_state=crate::checkpoint_state_marker::decode(&existing).unwrap_or_else(|e|panic!("{e}"));
                if old_state.sand.is_none() {
                    let current=bincode::serialize(&self.state).expect("accepted State serialize");
                    if old_state.payload!=current {panic!("CHECKPOINT_FRAME_IMMUTABLE: legacy grain checkpoint has changed; cannot overwrite");}
                    let old_history=crate::checkpoint_history::read_versioned(output,self.state.curr_frame,false)
                        .unwrap_or_else(|e|panic!("{e}"));
                    if let Some(old)=old_history {
                        let migration=match (&payload,&old) {
                            (crate::checkpoint_history::HistoryPayload::V2(next),crate::checkpoint_history::HistoryPayload::V1(prev)) =>
                                next.legacy.semantic_eq(prev) && !next.twist.prior_turns_known && next.mode.migration_time==Some(next.legacy.accepted_time),
                            _=>false,
                        };
                        if !migration && !payload.semantic_eq(&old){panic!("CHECKPOINT_FRAME_IMMUTABLE: accepted histories differ");}
                        if migration {
                            let old_bytes=std::fs::read(output.join(crate::checkpoint_commit::history_name(self.state.curr_frame))).expect("existing history read");
                            let (_,old_payload)=crate::checkpoint_commit::decode_history(self.state.curr_frame,&old_bytes).unwrap_or_else(|e|panic!("{e}"));
                            if crate::checkpoint_history::decode_payload(old_payload).unwrap_or_else(|e|panic!("{e}"))!=old {panic!("CHECKPOINT_FRAME_IMMUTABLE: migration history changed");}
                            crate::checkpoint_commit::publish_migration_history(output,self.state.curr_frame,&existing,&old_bytes,plastic.as_deref(),&history_bytes).unwrap_or_else(|e|panic!("{e}"));
                        } else {
                            history_bytes=old.encode().expect("existing checkpoint serialize");
                            crate::checkpoint_commit::publish(output,self.state.curr_frame,&existing,&history_bytes,plastic.as_deref()).unwrap_or_else(|e|panic!("{e}"));
                        }
                    } else {
                        if old_state.requirement!=crate::checkpoint_state_marker::HistoryRequirement::LegacyUnmarkedAmbiguous {
                            panic!("CHECKPOINT_HISTORY_REQUIRED: marked State requires runtime history");
                        }
                        let plastic_path=output.join(files::plastic_filename(self.state.curr_frame));
                        let old_plastic=match std::fs::symlink_metadata(&plastic_path) {
                            Ok(metadata) if metadata.file_type().is_file() => Some(std::fs::read(&plastic_path).expect("existing plastic read")),
                            Ok(_) => panic!("CHECKPOINT_FRAME_IMMUTABLE: plastic predecessor is not a regular file"),
                            Err(error) if error.kind()==std::io::ErrorKind::NotFound => None,
                            Err(error) => panic!("CHECKPOINT_FRAME_IMMUTABLE: plastic predecessor metadata: {error}"),
                        };
                        if old_plastic.as_deref()!=plastic.as_deref(){panic!("CHECKPOINT_FRAME_IMMUTABLE: plastic predecessor differs or missing");}
                    }
                    eprintln!("SAND_SPIN_LEGACY_COLD: unchanged legacy frame retained; grain spin remains unavailable");
                    return (existing,history_bytes,plastic,payload);
                }
                let n=mask.len().checked_mul(3).expect("CHECKPOINT_SAND_INVALID: size overflow");
                let omega=crate::driver::export_sand_checkpoint(n).unwrap_or_else(|e|panic!("{}",e.detail));
                let sand=crate::checkpoint_sand::encode(output,&self.state,mask,&omega).unwrap_or_else(|e|panic!("{e}"));
                state_bytes=crate::checkpoint_state_marker::bind_plastic(&crate::checkpoint_state_marker::encode_sand_state(&self.state,&sand).unwrap_or_else(|e|panic!("{e}")),plastic.as_deref()).unwrap_or_else(|e|panic!("{e}"));
                let old=crate::checkpoint_history::read_versioned(output,self.state.curr_frame,true).unwrap_or_else(|e|panic!("{e}")).expect("required history");
                let migration=match (&payload,&old) {
                    (crate::checkpoint_history::HistoryPayload::V2(next),crate::checkpoint_history::HistoryPayload::V1(prev)) =>
                        next.legacy.semantic_eq(prev) && !next.twist.prior_turns_known && next.mode.migration_time==Some(next.legacy.accepted_time),
                    _=>false,
                };
                if !migration && !payload.semantic_eq(&old){panic!("CHECKPOINT_FRAME_IMMUTABLE: accepted histories differ");}
                state_bytes=crate::checkpoint_state_marker::select_immutable_bytes(Some(&existing),&state_bytes).unwrap_or_else(|e|panic!("{e}")).into_owned();
                if migration {
                    let old_bytes=std::fs::read(output.join(crate::checkpoint_commit::history_name(self.state.curr_frame))).expect("existing history read");
                    let (_,old_payload)=crate::checkpoint_commit::decode_history(self.state.curr_frame,&old_bytes).unwrap_or_else(|e|panic!("{e}"));
                    if crate::checkpoint_history::decode_payload(old_payload).unwrap_or_else(|e|panic!("{e}"))!=old {panic!("CHECKPOINT_FRAME_IMMUTABLE: migration history changed");}
                    crate::checkpoint_commit::publish_migration_history(output,self.state.curr_frame,&state_bytes,&old_bytes,plastic.as_deref(),&history_bytes).unwrap_or_else(|e|panic!("{e}"));
                } else {
                    history_bytes=old.encode().expect("existing checkpoint serialize");
                    crate::checkpoint_commit::publish(output,self.state.curr_frame,&state_bytes,&history_bytes,plastic.as_deref()).unwrap_or_else(|e|panic!("{e}"));
                }
            }
            (state_bytes,history_bytes,plastic,payload)
        });
        // Existing State remains immutable; a validated V1 history may be augmented above.
        if transaction.is_some()&&Path::new(&program_args.output).join(files::state_filename(self.state.curr_frame)).exists(){crate::status_writer::note_saved();return;}
        let path_mesh = format!("{}/{}", program_args.output, files::MESHSET);
        let path_dataset = format!("{}/{}", program_args.output, files::DATASET);
        info!(">>> saving state started...");
        // A skipped dataset write may reuse only bytes already certified for this
        // positional layout. Validate the existing marker before invalidating it:
        // publishing a new marker must never upgrade unknown old dataset bytes.
        let dataset_exists = std::path::Path::new(&path_dataset).exists();
        if dataset_exists {
            crate::checkpoint_layout::validate(std::path::Path::new(&program_args.output))
                .unwrap_or_else(|e| panic!("{e}; checkpoint save refused; use a fresh output directory"));
        }
        // The dataset and the meshset are both build-time products that the
        // simulation does not change, so each is written once and skipped on
        // every later checkpoint. The single exception is the rest shape: the
        // dataset's `inv_rest*` matrices and rest angles are build-time values
        // like everything else here, but plasticity creeps them on the device
        // as the run proceeds. That is why they are ALSO written per frame,
        // and only for the kernels the scene enables. See `plastic_state`.
        if !dataset_exists {
            info!("saving dataset to {path_dataset}");
            super::save(dataset, path_dataset.as_str());

        }
        if !std::path::Path::new(&path_mesh).exists() {
            info!("saving meshset to {path_mesh}");
            super::save(&self.mesh, path_mesh.as_str());
        }
        if !dataset_exists {
            crate::checkpoint_layout::record(std::path::Path::new(&program_args.output));
        }
        if let Some(marker)=self.connected_collision.as_ref() {
            // Bind semantic/input/owner/pair/base/final-link identity to saved dataset/layout.
            crate::connected_collision_native::record_checkpoint(marker,Path::new(&program_args.output))
                .unwrap_or_else(|e|panic!("{e}"));
        }

        // Record which layout produced these checkpoints, so the resume path
        // knows whether the dataset's rest shape is the build-time one (this
        // layout, marker present) or the last frame's (a layout that rewrites
        // the dataset every save, marker absent). It must not be inferred from
        // the per-frame files themselves: they are pruned, so their absence
        // cannot distinguish "never written" from "deleted".
        let path_layout = format!("{}/{}", program_args.output, files::CHECKPOINT_LAYOUT);
        if !std::path::Path::new(&path_layout).exists() {
            std::fs::write(&path_layout, files::CHECKPOINT_LAYOUT_NOTE)
                .unwrap_or_else(|err| panic!("failed to write {path_layout}: {err}"));
        }
        // Before the state, deliberately. A checkpoint is resumable once its
        // `state_<N>.bin.gz` exists, so writing the rest shape first means a
        // crash between the two leaves a rest shape with no state, which is an
        // inert orphan, rather than a state with no rest shape, which is the
        // one combination that cannot be resumed.
        if let Some((mut state_bytes,history_bytes,plastic,_payload))=transaction {
            let output=Path::new(&program_args.output);
            if let Some(mask)=sand_mask.as_ref() {
                let n=mask.len().checked_mul(3).expect("CHECKPOINT_SAND_INVALID: size overflow");
                let omega=crate::driver::export_sand_checkpoint(n).unwrap_or_else(|e|panic!("{}",e.detail));
                let sand=crate::checkpoint_sand::encode(output,&self.state,mask,&omega).unwrap_or_else(|e|panic!("{e}"));
                state_bytes=crate::checkpoint_state_marker::bind_plastic(&crate::checkpoint_state_marker::encode_sand_state(&self.state,&sand).unwrap_or_else(|e|panic!("{e}")),plastic.as_deref()).unwrap_or_else(|e|panic!("{e}"));
            }
            crate::checkpoint_commit::publish(output,self.state.curr_frame,&state_bytes,&history_bytes,plastic.as_deref()).unwrap_or_else(|e|panic!("{e}"));
            let saved_before_pruning=ppf_cts_core::datamodel::list_saved_states(output);
            super::remove_old_files(&program_args.output,files::STATE_PREFIX,files::STATE_SUFFIX,sim_args.keep_states,self.state.curr_frame);
            super::remove_old_files(&program_args.output,files::PLASTIC_PREFIX,files::PLASTIC_SUFFIX,sim_args.keep_states,self.state.curr_frame);
            for frame in saved_before_pruning {if !output.join(files::state_filename(frame as i32)).exists(){
                std::fs::remove_file(output.join(crate::checkpoint_commit::history_name(frame as i32))).unwrap_or(());
            }}
            crate::status_writer::note_saved();return;
        }
        let plastic=self.checkpoint_plastic_bytes(dataset);
        let path_state = format!(
            "{}/{}",
            program_args.output,
            files::state_filename(self.state.curr_frame)
        );
        info!("saving state to {path_state}...");
        let state=crate::checkpoint_state_marker::encode_state(&self.state,false).expect("checkpoint State serialize");
        let state=crate::checkpoint_state_marker::bind_plastic(&state,plastic.as_deref()).expect("plastic binding");
        crate::checkpoint_commit::publish_without_history(Path::new(&program_args.output),self.state.curr_frame,&state,plastic.as_deref()).unwrap_or_else(|e|panic!("{e}"));
        super::remove_old_files(&program_args.output,files::PLASTIC_PREFIX,files::PLASTIC_SUFFIX,sim_args.keep_states,self.state.curr_frame);
        super::remove_old_files(
            &program_args.output,
            files::STATE_PREFIX,
            files::STATE_SUFFIX,
            sim_args.keep_states,
            self.state.curr_frame,
        );
        // A resumable checkpoint now exists on disk; reflect it in the
        // status record's `resumable` flag.
        crate::status_writer::note_saved();
        info!("<<< save state done.");
    }

    /// Write `plastic_<frame>.bin.gz`, this frame's rest shape, for a scene
    /// where plasticity creeps it away from the build-time values the dataset
    /// holds.
    ///
    /// Only the enabled kernels' arrays are pulled back from the device and
    /// only those are stored, so a scene with no plastic material does no DtoH
    /// copy and writes no file. Retention matches the state checkpoints, so a
    /// `state_<N>.bin.gz` that survives pruning still has its rest shape beside
    /// it.
    fn save_plastic_state(
        &self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        dataset: &DataSet,
    ) {
        use ppf_cts_formats::files;
        let kinds = PlasticKinds::of(dataset);
        if !kinds.any() {
            return;
        }
        // Refresh the host arrays from the device right before serializing: the
        // refresh is a real DOWNLOAD, not a copy out of a host-side mirror. The
        // plastic commit is a kernel, so the device holds the crept rest shape
        // and the host copy is stale from that dispatch until this runs. This
        // is the checkpoint path alone, which is where the shape has to reach
        // the host anyway.
        unsafe {
            if kinds.needs_inv_rest() {
                fetch_inv_rest();
            }
            if kinds.needs_rest_angles() {
                fetch_rest_angles();
            }
        }
        let path = format!(
            "{}/{}",
            program_args.output,
            files::plastic_filename(self.state.curr_frame)
        );
        info!("saving plastic rest shape to {path}");
        super::save(&PlasticState::extract(dataset, kinds), path.as_str());
        super::remove_old_files(
            &program_args.output,
            files::PLASTIC_PREFIX,
            files::PLASTIC_SUFFIX,
            sim_args.keep_states,
            self.state.curr_frame,
        );
    }

    fn checkpoint_plastic_bytes(&self,dataset:&DataSet)->Option<Vec<u8>>{
        use crate::plastic_state::{PlasticKinds,PlasticState};
        let kinds=PlasticKinds::of(dataset);if !kinds.any(){return None;}
        unsafe {if kinds.needs_inv_rest(){fetch_inv_rest();}if kinds.needs_rest_angles(){fetch_rest_angles();}}
        Some(crate::checkpoint_history::gzip(&PlasticState::extract(dataset,kinds)).expect("checkpoint plastic serialize"))
    }

    /// Hold the run at the current frame until the caller releases it.
    ///
    /// The process, the device and every buffer stay alive: this is a wait
    /// between two steps, not a checkpoint. Before the next step the solver
    /// takes whatever live inputs the caller rewrote while it was held.
    fn hold_at_frame(
        &self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        scene: &mut Scene,
        dataset: &DataSet,
    ) {
        use ppf_cts_formats::files;
        let output = std::path::Path::new(program_args.output.as_str());
        let frame = self.state.curr_frame;
        let held_path = output.join(files::HELD);
        let save_path = output.join(files::SAVE_AND_QUIT);
        std::fs::write(&held_path, format!("{frame}\n")).unwrap_or_else(|err| {
            panic!("cannot write {}: {err}", held_path.display())
        });
        info!("held at frame {frame}");
        let started = std::time::Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(20));
            if save_path.exists() {
                break;
            }
            match read_hold(&program_args.output) {
                None => break,
                Some((hold, timeout)) => {
                    if hold > frame {
                        break;
                    }
                    if started.elapsed() > timeout {
                        info!(
                            "held at frame {frame} for {:.0} s with no release; saving and \
                             quitting so the hold does not keep the device",
                            started.elapsed().as_secs_f64()
                        );
                        std::fs::write(&save_path, b"").unwrap();
                        break;
                    }
                }
            }
        }
        let updated = output.join(files::INPUTS_UPDATED);
        if updated.exists() && !save_path.exists() {
            scene.reload_dyn_param(&program_args.path);
            match crate::force_field::load(&program_args.path, dataset.vertex.curr.size as usize) {
                Ok(Some(field)) => {
                    info!("held run takes an updated {}", field.describe());
                    crate::driver::install_force_field(&field, sim_args.world_scaling);
                }
                Ok(None) => {
                    info!("held run takes updated inputs: the force field is removed");
                    crate::driver::install_force_field(
                        &crate::force_field::ForceFieldData::default(),
                        sim_args.world_scaling,
                    );
                }
                Err(message) => panic!("force field: {message}"),
            }
            std::fs::remove_file(&updated).unwrap_or_else(|err| {
                panic!("cannot remove {}: {err}", updated.display())
            });
        }
        let _ = std::fs::remove_file(&held_path);
        info!("released from frame {frame}");
    }

    pub fn run(
        &mut self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        mut dataset: DataSet,
        mut param: ParamSet,
        mut scene: Scene,
        restoring: Option<&crate::restoring_native::Installed>,
        restoring_seal: Option<&crate::restoring_curve::PreparedCurves>,
    ) {
        let initialize_finish_path = std::path::Path::new(program_args.output.as_str())
            .join(ppf_cts_formats::files::INITIALIZE_FINISH);
        let finished_path = std::path::Path::new(program_args.output.as_str())
            .join(ppf_cts_formats::files::FINISHED);
        if finished_path.exists() {
            std::fs::remove_file(finished_path.clone()).unwrap();
        }
        if initialize_finish_path.exists() {
            std::fs::remove_file(initialize_finish_path.clone()).unwrap();
        }
        let mut timeline_statistics = crate::statistics::TimelineStatistics::load(
            &program_args.path,
            &program_args.output,
            &self.mesh.mesh.mesh,
            sim_args.fps,
        );
        if let Some(statistics) = &timeline_statistics {
            statistics.configure_dataset(&mut dataset);
        }
        let joint_owned=scene.joint_input.as_ref().map(|input| {
            input.verify_unchanged(std::path::Path::new(&program_args.path)).unwrap_or_else(|e| panic!("joint handoff: {e}")); input.records().clone()
        });
        let soft_owned=scene.joint_input.as_ref().and_then(|input|
            crate::cross_links::prepare_batch(input,scene.args().world_scaling as f64).unwrap_or_else(|e|panic!("material/cross-link handoff: {e}")));
        let actual_fem_overlay=soft_owned.as_ref().map(crate::twist_overlay::fem_material_spin_from_owned).transpose().unwrap_or_else(|e|panic!("{e}")).flatten();
        if self.fem_material_spin_overlay.is_some() && self.fem_material_spin_overlay!=actual_fem_overlay {panic!("CHECKPOINT_MATERIAL_SPIN_INVALID: FEM selection changed");}
        self.fem_material_spin_overlay=actual_fem_overlay;
        let resuming=program_args.load>0;
        if !resuming||self.resume_history.is_some()||joint_owned.is_some()||restoring_seal.is_some()
            || crate::checkpoint_sand::mask(&dataset).unwrap_or_else(|e|panic!("{e}")).is_some() {
            let (uuid,identity)=joint_owned.as_ref().map(|o|(o.scene_uuid.clone(),o.scene_inputs_sha256.clone()))
                .or_else(||restoring_seal.map(|p|(p.snapshot.context.scene_uuid.clone(),p.snapshot.context.scene_inputs_sha256.clone())))
                .unwrap_or_default();
            self.checkpoint_context=Some((uuid,identity,std::fs::read(Path::new(&program_args.path).join("param.toml")).expect("sealed parameters")));
        }
        // New material exclusions reconstruct exclusively from retained sealed base links.
        // All original subsystem resume gates remain in main; no feature bypass.
        let connected=soft_owned.as_ref().and_then(|s|s.connected_collision.as_ref());
        let filtered=if let Some(prepared)=connected {
            let marker=if resuming{self.connected_collision.clone().ok_or_else(||"BONE_CONNECTED_CHECKPOINT_UNKNOWN: resume was not prevalidated".to_owned())}else{crate::connected_collision_native::install(&mut dataset,prepared,program_args.load,Path::new(&program_args.output))}
                .unwrap_or_else(|e|panic!("{e}"));
            let count=marker.pair_count;
            self.connected_collision=Some(marker);
            count
        } else if resuming {self.resume_history.as_ref().map_or(0,|h|h.filtered_links)}
        else {joint_owned.as_ref().map_or(0,|owned| unsafe { crate::native_joint::filter_connected(&mut dataset,owned) })};
        self.checkpoint_filtered=filtered;
        let mut joint_accepted=self.resume_history.as_ref().map_or(0,|h|h.joint_accepted);
        self.checkpoint_accepted=joint_accepted;
        if let Some(seal)=restoring_seal {
            // Final snapshot gate immediately before dataset initialization.
            seal.verify_unchanged(std::path::Path::new(&program_args.path)).unwrap_or_else(|e|panic!("restoring handoff: {e}"));
        }
        let mut collision_windows_enabled=false;
        if unsafe { initialize(&dataset, &param) } {
            if sim_args.pcg_precision != crate::args::PcgPrecision::Fp32 {
                crate::driver::enable_mixed_pcg(sim_args.pcg_precision)
                    .unwrap_or_else(|fatal| panic!("{}", fatal.detail));
                info!("PCG_PRECISION: {:?} (FP32 true residual)", sim_args.pcg_precision);
            }
            if let Some(owned)=joint_owned.as_ref() {
                crate::driver::install_joints_from_rest(owned,self.sealed_initial.as_deref());
                crate::driver::install_hard_joints(&owned.hard_records,owned.payload_version==5,
                    owned.roots.len(),dataset.translation_lock.size as usize)
                    .unwrap_or_else(|fatal|panic!("hard joint install: {}",fatal.detail));
                crate::native_joint::write_receipt(&program_args.output,owned,0,self.state.time,"INITIALIZED",filtered);
            }
            if let Some(owned)=soft_owned.as_ref() {
                crate::driver::install_soft_joints_from_rest(owned,resuming).unwrap_or_else(|fatal|panic!("FEM joint install: {}",fatal.detail));
                crate::soft_joint::write_receipt(std::path::Path::new(&program_args.output),owned,0,self.state.time,"INITIALIZED").expect("FEM joint receipt");
            }
            if let Some(curves)=restoring {
                if let Some(seal)=restoring_seal {seal.verify_unchanged(std::path::Path::new(&program_args.path)).unwrap_or_else(|e|panic!("restoring upload seal: {e}"));}
                crate::driver::install_restoring(curves,dataset.constraint.stitch.size as usize)
                    .unwrap_or_else(|fatal|panic!("restoring install: {}",fatal.detail));
            }
            // Establish the trusted selected-body fit policy before any actual
            // pose sampling or restored-angle validation.
            if let Some(ctx)=&self.twist_checkpoint_context {
                if ctx.mask.len()!=ctx.joints.len() || ctx.joints.iter().zip(&ctx.mask).filter_map(|(id,on)|on.then_some(id.clone())).collect::<Vec<_>>()!=ctx.selected {
                    panic!("CHECKPOINT_TWIST_INVALID: trusted mask/order mismatch");
                }
            }
            if self.twist_checkpoint_context.is_some()||self.material_spin_overlay.is_some()||self.fem_material_spin_overlay.is_some() {
                crate::driver::prepare_material_fit_selections(self.twist_checkpoint_context.as_ref().map(|c|c.mask.as_slice()),self.material_spin_overlay.as_ref().map(|c|c.mask.as_slice()),self.fem_material_spin_overlay.as_ref().map(|c|c.mask.as_slice())).unwrap_or_else(|e|panic!("material fit policy: {}",e.detail));
            }
            if let Some(omega)=self.resume_sand.take() {
                crate::driver::import_sand_checkpoint(&omega).unwrap_or_else(|e|panic!("sand device import: {}",e.detail));
                info!("SAND_SPIN_DEVICE_MATCHED frame={} time_bits={:016x} floats={}",program_args.load,self.state.time.to_bits(),omega.len());
            }
            if let Some(history)=&self.resume_history {
                crate::driver::import_checkpoint(&history.runtime,self.state.time).unwrap_or_else(|e|panic!("checkpoint runtime import: {}",e.detail));
                let actual=crate::driver::export_checkpoint(self.state.time).unwrap_or_else(|e|panic!("checkpoint actual readback: {}",e.detail));
                if !actual.semantic_eq(&history.runtime){panic!("CHECKPOINT_RESTORE_ROUNDTRIP_MISMATCH: accepted runtime differs after actual device import/readback");}
                self.checkpoint_adaptive=Some(history.adaptive.clone());self.checkpoint_adaptive_start=history.adaptive_start_time;
                let receipt=serde_json::json!({"kind":"ppf.bone_checkpoint.restore_roundtrip","version":1,"state":"MATCHED","restored_frame":program_args.load,
                    "accepted_time_seconds":self.state.time,"launch_id":crate::status_writer::launch_id(),"comparison":"exact_scalar_bytes_excluding_new_process_attempt_identity",
                    "ignored_fields":["root.source_generation","fem.source_launch_id"],"device_running_rotation_f32_bits":actual.running_rotation.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                    "device_curve_history":actual.restoring.as_ref().map(|r|&r.history),"restoring_commit_count":actual.restoring_commit_count,"restoring_commit_time":actual.restoring_commit_time,
                    "fem_accepted_steps":actual.fem_steps,"root_and_fem_accepted_history_equal":true,"adaptive_controller":self.checkpoint_adaptive,"adaptive_start_time_seconds":self.checkpoint_adaptive_start});
                let path=Path::new(&program_args.output).join("restore_roundtrip_native_receipt.json");let tmp=path.with_extension("json.tmp");
                std::fs::write(&tmp,serde_json::to_vec_pretty(&receipt).expect("restore roundtrip receipt serialize")).expect("restore roundtrip receipt stage");std::fs::rename(tmp,path).expect("restore roundtrip receipt publish");
            }
            if let (Some(ctx),Some(crate::checkpoint_history::HistoryPayload::V3(v)))=(&self.material_spin_overlay,&self.resume_twist_payload) {
                v.validate_material(&ctx.scene_uuid,&ctx.selected_joint_uuids).unwrap_or_else(|e|panic!("{e}"));
                let accepted=v.rigid_material.clone().expect("CHECKPOINT_MATERIAL_SPIN_INVALID: missing rigid history");
                let q=crate::driver::sample_committed_material_spin(&ctx.mask).unwrap_or_else(|e|panic!("material spin restored pose: {}",e.detail));
                for (actual,row) in q.iter().zip(&ctx.mask).filter_map(|(q,enabled)|enabled.then_some(q)).zip(&accepted.rows) {
                    let dot:f64=actual.iter().zip(row.q_wxyz).map(|(a,b)|a*b).sum();
                    if !dot.is_finite() || (1.-dot.abs()).abs()>1e-4 {panic!("CHECKPOINT_MATERIAL_SPIN_INVALID: restored orientation mismatch");}
                }
                crate::driver::activate_material_spin(accepted,&ctx.scene_uuid,&ctx.selected_joint_uuids,self.state.time,ctx.mask.clone()).unwrap_or_else(|e|panic!("material spin restore: {}",e.detail));
            }
            if let (Some(ctx),Some(crate::checkpoint_history::HistoryPayload::V3(v)))=(&self.fem_material_spin_overlay,&self.resume_twist_payload) {
                let accepted=v.fem_material.clone().expect("CHECKPOINT_MATERIAL_SPIN_INVALID: missing FEM history");
                accepted.validate(&ctx.scene_uuid,&ctx.selected_joint_uuids,self.state.time).unwrap_or_else(|e|panic!("{e}"));
                let actual=crate::driver::sample_committed_fem_material_spin(&ctx.mask).unwrap_or_else(|e|panic!("FEM material spin restored pose: {}",e.detail));
                if actual.len()!=accepted.rows.len(){panic!("CHECKPOINT_MATERIAL_SPIN_INVALID: FEM pose count mismatch");}
                for (actual,row) in actual.iter().zip(&accepted.rows){let dot:f64=actual.q_wxyz.iter().zip(row.q_wxyz).map(|(a,b)|a*b).sum();if !dot.is_finite()||(1.-dot.abs()).abs()>1e-4{panic!("CHECKPOINT_MATERIAL_SPIN_INVALID: FEM restored orientation mismatch");}}
                crate::driver::activate_fem_material_spin(accepted,&ctx.scene_uuid,&ctx.selected_joint_uuids,self.state.time,ctx.mask.clone()).unwrap_or_else(|e|panic!("FEM material spin restore: {}",e.detail));
            }
            if let Some(payload)=&self.resume_twist_payload {
                if let Some(ctx)=&self.twist_checkpoint_context {
                    let principal=ctx.principal();
                    let migration=if ctx.explicit_migration {Some(ctx.mode.clone())}else{None};
                    let restored=payload.restore_twist(&ctx.scene,&ctx.joints,&principal,migration,&ctx.sealed_models,&ctx.selected).unwrap_or_else(|e|panic!("{e}"));
                    let (accepted,mode)=restored.expect("CHECKPOINT_TWIST_INVALID: continuous mode missing history");
                    crate::driver::import_accepted_twist(accepted,&ctx.scene,&ctx.joints,self.state.time).unwrap_or_else(|e|panic!("checkpoint twist import: {}",e.detail));
                    crate::driver::activate_twist_selection(ctx.mask.clone()).unwrap_or_else(|e|panic!("checkpoint twist activation: {}",e.detail));
                    self.twist_checkpoint_context.as_mut().expect("twist context").mode=mode;
                    self.twist_checkpoint_context.as_mut().expect("twist context").explicit_migration=false;
                } else if matches!(payload,crate::checkpoint_history::HistoryPayload::V2(_)) || matches!(payload,crate::checkpoint_history::HistoryPayload::V3(v) if v.twist.is_some()) {
                    panic!("CHECKPOINT_TWIST_UNAVAILABLE: trusted restore contract not installed");
                }
            }
            if !resuming {
                if let Some(ctx)=&self.fem_material_spin_overlay {
                    let rows=crate::driver::sample_committed_fem_material_spin(&ctx.mask).unwrap_or_else(|e|panic!("FEM material spin initial pose: {}",e.detail));
                    if rows.len()!=ctx.selected_joint_uuids.len()||rows.iter().any(|r|r.q_wxyz[1..].iter().map(|v|v*v).sum::<f64>().sqrt()>1e-4){panic!("FEM_MATERIAL_SPIN_INITIAL_INVALID: fresh run must start at declared rest");}
                    let accepted=crate::material_spin_history::AcceptedMaterialSpin{scene_uuid:ctx.scene_uuid.clone(),joint_uuids:ctx.selected_joint_uuids.clone(),time:self.state.time,generation:0,rows};
                    crate::driver::activate_fem_material_spin(accepted,&ctx.scene_uuid,&ctx.selected_joint_uuids,self.state.time,ctx.mask.clone()).unwrap_or_else(|e|panic!("FEM material spin initialize: {}",e.detail));
                }
                if let Some(ctx)=&self.material_spin_overlay {
                    let q=crate::driver::sample_committed_material_spin(&ctx.mask).unwrap_or_else(|e|panic!("material spin initial pose: {}",e.detail));
                    let rows=q.into_iter().zip(&ctx.mask).filter_map(|(mut q,enabled)|if *enabled {
                        let n=q.iter().map(|x|x*x).sum::<f64>().sqrt();
                        if !n.is_finite() || n<0.99 || n>1.01 {panic!("MATERIAL_SPIN_INITIAL_INVALID: nonunit pose");}
                        for x in &mut q {*x/=n;}
                        if q[1..].iter().map(|x|x*x).sum::<f64>().sqrt()>1e-4 {panic!("MATERIAL_SPIN_INITIAL_INVALID: fresh run must start at declared rest");}
                        Some(crate::material_spin_history::MaterialSpinRow{q_wxyz:q,a:[0.;3]})
                    }else{None}).collect();
                    let accepted=crate::material_spin_history::AcceptedMaterialSpin{scene_uuid:ctx.scene_uuid.clone(),joint_uuids:ctx.selected_joint_uuids.clone(),time:self.state.time,generation:0,rows};
                    crate::driver::activate_material_spin(accepted,&ctx.scene_uuid,&ctx.selected_joint_uuids,self.state.time,ctx.mask.clone()).unwrap_or_else(|e|panic!("material spin initialize: {}",e.detail));
                }
                if let Some(ctx)=&self.twist_checkpoint_context {
                    let theta=ctx.principal();
                    let history=crate::continuous_twist::AcceptedTwist {scene_uuid:ctx.scene.clone(),joint_uuids:ctx.joints.clone(),time:self.state.time,theta,prior_turns_known:true};
                    history.validate(&ctx.scene,&ctx.joints,self.state.time).unwrap_or_else(|e|panic!("{e}"));
                    crate::continuous_twist::project_device(&history.theta).unwrap_or_else(|e|panic!("{e}"));
                    crate::driver::import_accepted_twist(history,&ctx.scene,&ctx.joints,self.state.time).unwrap_or_else(|e|panic!("checkpoint twist initialize: {}",e.detail));
                    crate::driver::activate_twist_selection(ctx.mask.clone()).unwrap_or_else(|e|panic!("checkpoint twist activation: {}",e.detail));
                }
            }
            // Destructive tail cleanup is deferred until both static preflight
            // and this real import/readback gate have succeeded.
            if resuming {crate::prune_output_prefix(program_args);}
            // Initialize collision windows on GPU
            let cw = scene.build_collision_window_table();
            collision_windows_enabled=cw.has_windows;
            info!("collision windows: has_windows={}, n_groups={}, vert_dmap_len={}, window_counts={:?}",
                  cw.has_windows, cw.n_groups, cw.vert_dmap.len(),
                  if cw.has_windows { &cw.window_counts[..cw.n_groups.min(20) as usize] } else { &[] });
            if cw.has_windows {
                unsafe {
                    init_collision_windows(
                        cw.vert_dmap.as_ptr(),
                        cw.vert_dmap.len() as u32,
                        cw.windows.as_ptr(),
                        cw.window_counts.as_ptr(),
                        cw.n_groups,
                    );
                }
            }
            // THE EXTERNAL FORCE FIELD, read, verified and installed once. A
            // malformed field stops the run here, naming the file and the
            // field, rather than reaching a dispatch; a session with none
            // installs nothing and every step runs as it always has.
            match crate::force_field::load(
                &program_args.path,
                dataset.vertex.curr.size as usize,
            ) {
                Ok(Some(field)) => {
                    info!("{}", field.describe());
                    for script in &field.scripts {
                        info!("force field script:\n{}", script.source);
                    }
                    crate::driver::install_force_field(&field, sim_args.world_scaling);
                }
                Ok(None) => {}
                Err(message) => panic!("force field: {message}"),
            }
            write_current_time_to_file(initialize_finish_path.to_str().unwrap()).unwrap();
            let _status_phase = crate::driver::phase::start("solver.host_status_progress");
            crate::status_writer::progress(
                Phase::Initialized,
                self.state.curr_frame.max(0),
                self.state.time,
            );
            drop(_status_phase);
        } else {
            // initialize() reports failure through a bare `false`, so the
            // cause has to come from the fatal code the backend stamped
            // alongside it. A code of zero means a failure path inside
            // initialize() that named no cause; report that as unknown rather
            // than inheriting the label of the one path that does set a code,
            // which would attribute a future second failure mode to an
            // intersection it never saw.
            crate::status_writer::terminal_init_failure();
            std::process::exit(1);
        }
        if program_args.load > 0 && !sim_args.disable_contact {
            unsafe {
                update_dyn(
                    self.state.dyn_index.as_ptr(),
                    self.state.dyn_offset.as_ptr(),
                );
            }
        }
        let mut frame_publisher = if program_args.async_output { Some(crate::frame_output::Publisher::new(&program_args.output)) } else { None };
        let mut last_time = Instant::now();

        // Frame 0 is the rest pose at sim time 0. It is written once
        // before the loop so the loop body has a single concern: each
        // iteration prepares the *target* pin positions for the next
        // advance, runs the step, and writes vert_n.bin only after the
        // step completes, so the pinned and the free vertices in one
        // file always reflect the same sim time.
        //
        // THE ORDER IS WHAT MAKES THAT TRUE, and it is easy to lose.
        // Setting the pin constraint from `pin_at(state.time)`, the START
        // of the step, and writing vert_n.bin BEFORE the next advance puts
        // step n's pinned positions at the previous iteration's start time
        // while the free vertices in the same file have been integrated to
        // `state.time`. The two halves of one file then disagree by one
        // step, which is what the `bl_pin_*` fidelity scenarios detect.
        if self.state.curr_frame < 0 && self.state.time == 0.0 {
            self.write_frame_outputs(
                program_args,
                sim_args,
                &dataset,
                &param,
                &scene,
                &mut last_time,
                0,
                0.0,
                0.0,
                0.0,
                &mut timeline_statistics,
                0,
                &mut frame_publisher,
                true,
            );
            self.state.curr_frame = 0;
            {
                let _status_phase = crate::driver::phase::start("solver.host_status_progress");
                if frame_publisher.is_none() { crate::status_writer::progress(Phase::Running, 0, self.state.time); }
            }
        }
        // Precompute the per-keyframe inverse rest matrices for a streamed
        // time-varying rest shape (the captured pull-pin deformation). Reuses
        // the exact build-time formula (builder::compute_inv_rest) so init and
        // animated rest shapes stay identical, then interpolates between
        // keyframes per step. Empty (no per-step upload) when the scene has no
        // schedule.
        // Assembled lazily on the first step: `self.mesh` is borrowed
        // immutably to build it and the surrounding scope holds `&mut self`,
        // so it is cheaper to defer than to restructure the borrow here.
        let mut material_keyframes: Option<Vec<MaterialKeyframe>> = None;
        let rest_shape_keyframes: Vec<RestShapeKeyframe> = match scene.rest_shape_schedule() {
            Some((times, frames)) => {
                let face_props = dataset.prop.face.as_slice();
                let tet_props = dataset.prop.tet.as_slice();
                let face_params = dataset.param_arrays.face.as_slice();
                let tet_params = dataset.param_arrays.tet.as_slice();
                times
                    .into_iter()
                    .zip(frames.iter())
                    .map(|(t, rest_v)| {
                        // exclude_singular = true: a folded captured frame can
                        // drive a rest element near-singular; rather than
                        // clamp (which fattens it and hides the bend's
                        // stretch), flag it so update_rest_shape drops it from
                        // the energy. Every non-singular element keeps its
                        // exact rest.
                        let (inv2, inv3, ef, et) = super::builder::compute_inv_rest(
                            rest_v,
                            &self.mesh,
                            face_props,
                            tet_props,
                            face_params,
                            tet_params,
                            true,
                        );
                        (t, inv2, inv3, ef, et)
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        // CUDA-only optional telemetry. Windows keeps one NVML connection for
        // this run, avoiding a process launch on every physics step.
        #[cfg(cuda_backend)]
        let mut gpu_clock = {
            let _clock_init_phase = crate::driver::phase::start("solver.host_clock_init");
            crate::gpu_clock::Sampler::new()
        };
        let mut adaptive_controller = self.checkpoint_adaptive.take().unwrap_or_else(||crate::driver::adaptive_controller::Controller::new(param.dt));
        self.checkpoint_adaptive=Some(adaptive_controller.clone());
        self.checkpoint_adaptive_start=self.resume_history.as_ref().map_or(0.,|h|h.adaptive_start_time);
        let requested_frames=program_args.target_frame.unwrap_or(sim_args.frames);
        let adaptive_configuration = joint_owned.as_ref().map_or("unsealed", |owned| owned.scene_inputs_sha256.as_str()).to_owned();
        let mut retry_save_latched=false;
        loop {
            if let Some(p)=frame_publisher.as_ref() { p.check().expect("frame publisher before advance"); }
            #[cfg(cuda_backend)]
            let clock_sample = {
                let _clock_sample_phase = crate::driver::phase::start("solver.host_clock_sample");
                gpu_clock.sample()
            };
            #[cfg(cuda_backend)]
            if let Some(clock) = clock_sample {
                let _clock_log_phase = crate::driver::phase::start("solver.host_clock_log");
                info!("GPU SM Clock: {clock} MHz");

                // Name: GPU SM Clock Speed
                // Format: list[(time, MHz)]
                // Description:
                // Fresh SM clocks for all physical NVIDIA GPUs in index order,
                // retaining the original unfiltered query's representation.
                // Windows queries NVML every step; other CUDA platforms query
                // nvidia-smi at most once per wall second. No cached samples
                // are emitted with a later simulation timestamp.
                /*== push "clock" ==*/
                if let Ok(mut clock_log) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{}/data/clock.out", program_args.output).as_str())
                {
                    // Clock telemetry is optional: a log failure cannot abort
                    // otherwise valid simulation output.
                    let _ = writeln!(clock_log, "{} {}", self.state.time, clock);
                }
            }

            let _control_phase = crate::driver::phase::start("solver.host_control_before_advance");
            // THE FRAME-STEP HOLD, checked before save_and_quit so a caller
            // can save a held run. A run past its last frame is not held: it
            // finishes, which is what a hold beyond the end means.
            if self.state.curr_frame < requested_frames
                && !retry_save_latched
                && !Path::new(&program_args.output).join("save_and_quit").exists()
            {
                if let Some((hold, _)) = read_hold(&program_args.output) {
                    if self.state.curr_frame >= hold {
                        if let Some(p)=frame_publisher.as_mut() { p.barrier().expect("drain frames before hold"); }
                        self.hold_at_frame(program_args, sim_args, &mut scene, &dataset);
                    }
                }
            }

            // save_and_quit / frames-done checks happen at the top now;
            // the previous flow placed them after the frame-write
            // block, but with the write moved post-advance these need
            // to gate the *upcoming* step instead.
            let save_and_quit_path =
                std::path::Path::new(program_args.output.as_str()).join("save_and_quit");
            let save_requested = retry_save_latched || save_and_quit_path.exists();
            let checkpoint_path = Path::new(&program_args.output)
                .join(ppf_cts_formats::files::state_filename(self.state.curr_frame));
            let wait_for_frame = if save_requested && checkpoint_path.exists() {
                let bytes = std::fs::read(&checkpoint_path).expect("existing checkpoint read");
                let decoded = crate::checkpoint_state_marker::decode(&bytes)
                    .unwrap_or_else(|e| panic!("{e}"));
                let saved: State = bincode::deserialize(&decoded.payload)
                    .expect("existing checkpoint State decode");
                if !saved.time.is_finite() || !param.time.is_finite() || saved.time > param.time {
                    panic!("CHECKPOINT_CLOCK_INVALID: existing checkpoint time is invalid");
                }
                saved.time < param.time
            } else {
                false
            };
            if save_requested && !wait_for_frame {
                if let Some(p)=frame_publisher.as_mut() { p.finish().expect("finish frame publisher before save and quit"); }
                info!("save_and_quit file found, saving state...");
                drop(_control_phase);
                let _status_phase = crate::driver::phase::start("solver.host_status_progress");
                crate::status_writer::progress(
                    Phase::Saving,
                    self.state.curr_frame.max(0),
                    self.state.time,
                );
                drop(_status_phase);
                let _control_phase = crate::driver::phase::start("solver.host_control_before_advance");
                self.save_state(program_args, sim_args, &dataset,&param);
                std::fs::remove_file(save_and_quit_path).unwrap_or_else(|err| {
                    println!("Failed to delete 'save_and_quit' file: {err}");
                });
                crate::status_writer::terminal(Outcome::SavedAndQuit);
                break;
            }

            if self.state.curr_frame >= requested_frames {
                if let Some(p)=frame_publisher.as_mut() { p.finish().expect("finish frame publisher before FINISHED"); }
                if sim_args.auto_save > 0 || sim_args.save_state_on_finish {
                    info!("simulation finished, saving state...");
                    self.save_state(program_args, sim_args, &dataset,&param);
                } else {
                    info!("simulation finished, not saving state...");
                }
                if let Some(owned)=joint_owned.as_ref() { crate::native_joint::write_receipt(&program_args.output,owned,joint_accepted,self.state.time,"FINISHED",filtered); }
                if let Some(owned)=soft_owned.as_ref() { crate::soft_joint::write_receipt(std::path::Path::new(&program_args.output),owned,joint_accepted,self.state.time,"FINISHED").expect("FEM joint receipt"); }
                crate::status_writer::terminal(Outcome::Finished);
                break;
            }

            // Precompute the material keyframes on the first step. The
            // face table is expanded to one entry per face for an animated
            // scene, so this is the largest per-frame upload here and it is
            // assembled once per keyframe rather than once per step.
            if material_keyframes.is_none() {
                material_keyframes = Some(if scene.has_material_animation() {
                    let face_area =
                        crate::triutils::face_areas(&self.mesh.vertex, &self.mesh.mesh.mesh.face);
                    let tet_volumes =
                        crate::triutils::tet_volumes(&self.mesh.vertex, &self.mesh.mesh.mesh.tet);
                    scene
                        .param_anim_times()
                        .iter()
                        .map(|&t| {
                            let props =
                                scene.make_props_at(t, &self.mesh, &face_area, &tet_volumes);
                            // The hinge, edge and vertex tables are averages of
                            // the faces, so this frame's faces imply this
                            // frame's derived tables. Deriving them here, once
                            // per keyframe, keeps the per-step work to a blend.
                            let (vertex, edge, hinge) = crate::builder::rederive_animated_tables(
                                &self.mesh,
                                dataset.prop.face.as_slice(),
                                &props.face_params,
                                dataset.prop.edge.as_slice(),
                                dataset.prop.hinge.as_slice(),
                                dataset.prop.vertex.as_slice(),
                                dataset.param_arrays.vertex.as_slice(),
                                dataset.param_arrays.edge.as_slice(),
                                &face_area,
                            );
                            MaterialKeyframe {
                                time: t,
                                face: props.face_params,
                                vertex,
                                edge,
                                hinge,
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                });
            }

            // Interpolate dynamic params (including a keyframed dt) for this
            // step BEFORE deriving the step's target time. update_param looks
            // up self.state.time (the step START), which is the correct sample
            // point for the dt that governs this step, and it touches neither
            // the pin constraint nor the rest shape. Doing it first means
            // make_constraint, interp_rest_shape, and advance all use the same,
            // freshly-interpolated param.dt; otherwise a dt keyframe would land
            // one step late, targeting pins/rest at the previous step's dt.
            let adaptive_retry = adaptive_cg_retry_enabled();
            let host_fem_retry = soft_owned.as_ref().is_some_and(|o|!o.records.is_empty());
            let host_material_retry = self.material_spin_overlay.is_some();
            // Only an installed, selected continuous scene can opt into the
            // independent swing-sample host route. Its live state and sampled
            // playback are checked again inside the logical step below.
            let host_swing_candidate = !host_fem_retry && self.twist_checkpoint_context.as_ref().is_some_and(|ctx|ctx.mask.iter().any(|v|*v));
            #[cfg(all(test, feature="cpu"))]
            backend_save_fixture::capture_if_ready(self, &dataset, &param);
            let logical_start_time = self.state.time;
            let cg_limit = param.cg_max_iter;
            let cg_tolerance = param.cg_tol;
            let retry_run = crate::driver::adaptive_controller::logical_step(
                adaptive_retry, host_fem_retry||host_material_retry, &mut adaptive_controller, &mut param,
                logical_start_time, &adaptive_configuration, cg_limit, cg_tolerance,
                |param| scene.update_param(sim_args, logical_start_time, param),
                |param, selection| {
            let nominal_h = selection.authored_nominal_h;
            let host_swing_retry=host_swing_candidate && crate::driver::host_swing_retry_scope(param.playback)
                .unwrap_or_else(|fatal|crate::driver::report_host_fatal(fatal));
            let host_newton_root_retry=host_material_retry && rest_shape_keyframes.is_empty() && crate::driver::host_newton_root_scope(param.playback)
                .unwrap_or_else(|fatal|crate::driver::report_host_fatal(fatal));
            let host_kinematics_retry=(scene.has_time_sensitive_inputs() || collision_windows_enabled)
                && !host_fem_retry && !host_swing_retry && !host_material_retry
                && crate::driver::host_kinematics_retry_scope(param.playback).unwrap_or_else(|fatal|crate::driver::report_host_fatal(fatal));
            let host_domain_retry=host_fem_retry||host_swing_retry||host_material_retry||host_kinematics_retry;
            let seed_snapshot = std::rc::Rc::new(std::cell::RefCell::new(AdaptiveSeedSnapshotGuard::new()));
            let attempt_seed = seed_snapshot.clone();
            let finish_seed = seed_snapshot.clone();
            let host_seed_snapshot=std::rc::Rc::new(std::cell::RefCell::new(HostSeedSnapshotGuard::new()));
            let host_attempt_seed=host_seed_snapshot.clone();let host_finish_seed=host_seed_snapshot.clone();
            let domain_reprepare=std::cell::Cell::new(false);
            let mut prepare = |candidate_h, _retry_index, continuation, remaining_rejections| -> Result<crate::driver::HostAdvanceOutcome,crate::driver::HostFatal> {
            if host_newton_root_retry && continuation!=0 && save_and_quit_path.exists() {
                host_attempt_seed.borrow_mut().capture_or_restore()?;
                #[cfg(all(test, feature="cpu"))] backend_save_fixture::observe_save_request();
                return Ok(crate::driver::HostAdvanceOutcome::SaveRequested{continuation});
            }
            param.dt = candidate_h;
            #[cfg(all(test, feature="cpu"))]
            let private_old_cap = backend_save_fixture::cap_if_armed(continuation, param);
            let prepared=if host_kinematics_retry {
                let boundary=scene.next_discrete_input_event(self.state.time);
                Some(unsafe{crate::driver::prepare_kinematics_plan(continuation,candidate_h,boundary)}?)
            }else if domain_reprepare.get(){Some(unsafe{crate::driver::prepare_host_plan(continuation,candidate_h)}?)}else if host_newton_root_retry {Some(unsafe{crate::driver::prepare_initial_newton_root_plan(candidate_h)}?)}else{None};
            let integration_h=prepared.map_or(candidate_h,|p|p.integration_h());
            let _attempt_control_phase = crate::driver::phase::start("solver.host_control_before_advance");
            if host_domain_retry {
                host_attempt_seed.borrow_mut().capture_or_restore()?;
            } else if adaptive_retry {
                unsafe { attempt_seed.borrow_mut().capture_or_restore() };
            }

            prepare_attempt_inputs(&scene, self.state.time, candidate_h, param.playback,
                prepared, integration_h, param.prev_dt, soft_owned.as_ref().is_some_and(|o|o.control_events.is_some()),
                host_domain_retry, collision_windows_enabled, host_kinematics_retry, &rest_shape_keyframes,
                material_keyframes.as_deref())?;
            #[cfg(all(test, feature="cpu"))]
            backend_save_fixture::override_prev_if_armed(continuation, integration_h, &dataset)?;
            drop(_attempt_control_phase);
            let _phase = crate::driver::phase::start("solver.advance");
            if let Some(p)=frame_publisher.as_ref() { p.check().expect("frame publisher immediately before advance"); }
            let outcome = if host_kinematics_retry {
                unsafe { crate::driver::advance_host_kinematics(continuation,remaining_rejections,adaptive_retry,prepared) }
            } else if host_fem_retry || host_material_retry {
                unsafe { crate::driver::advance_host(continuation,remaining_rejections,adaptive_retry,true,prepared) }
            } else if host_swing_retry {
                unsafe { crate::driver::advance_host_swing(continuation,remaining_rejections,adaptive_retry,prepared) }
            } else {
                let mut result=StepResult::default();
                let returned=if adaptive_retry {unsafe{advance_retry(&mut result,continuation,remaining_rejections)}}else{unsafe{advance(&mut result)};0};
                if result.success(){Ok(crate::driver::HostAdvanceOutcome::Accepted(result))}
                else if result.retry_reason==crate::data::RETRY_REASON_PCG_MAX_ITERATIONS {Ok(crate::driver::HostAdvanceOutcome::PcgCap{result,continuation:returned,failed_dt:candidate_h})}
                else {Ok(crate::driver::HostAdvanceOutcome::Refused(result))}
            };
            #[cfg(all(test, feature="cpu"))]
            backend_save_fixture::after_advance(private_old_cap, param, &outcome, &save_and_quit_path);
            if matches!(&outcome,Ok(crate::driver::HostAdvanceOutcome::FemMaterialRejected{..}|crate::driver::HostAdvanceOutcome::SwingSampleRejected{..}|crate::driver::HostAdvanceOutcome::ChartPathRejected{..}|crate::driver::HostAdvanceOutcome::MaterialSpinRejected{..}|crate::driver::HostAdvanceOutcome::NewtonCapRejected{..})) || (host_newton_root_retry && matches!(&outcome,Ok(crate::driver::HostAdvanceOutcome::PcgCap{..}))){domain_reprepare.set(true);}
            drop(_phase);
            outcome
            };
            let run = if host_domain_retry {
                crate::driver::adaptive_retry::run_host_attempts(&mut prepare,|token| { let r=crate::driver::cancel_host_retry(token); #[cfg(all(test, feature="cpu"))] if r.is_ok(){backend_save_fixture::observe_cancel();} r },
                    move || {let r=host_finish_seed.borrow_mut().finish(); #[cfg(all(test, feature="cpu"))] if r.is_ok(){backend_save_fixture::observe_finish();} r},selection.initial_trial_h,logical_start_time,
                    crate::driver::adaptive_retry::MAX_REJECTIONS,selection.controller_ceiling_h)
                    .unwrap_or_else(|fatal|crate::driver::report_host_fatal(fatal))
            } else {
                let mut retry_ops = crate::driver::adaptive_retry::ClosureRetryAttemptOps::new(
                    |h,index,token,budget| match prepare(h,index,token,budget).unwrap_or_else(|fatal|crate::driver::report_host_fatal(fatal)) {
                        crate::driver::HostAdvanceOutcome::KinematicsShortened{..}=>unreachable!("C ABI cannot return a kinematics retry"),
                        crate::driver::HostAdvanceOutcome::Accepted(result)|crate::driver::HostAdvanceOutcome::Refused(result)=>crate::driver::adaptive_retry::RetryAttempt{result,continuation:0},
                        crate::driver::HostAdvanceOutcome::PcgCap{result,continuation,..}=>crate::driver::adaptive_retry::RetryAttempt{result,continuation},
                        crate::driver::HostAdvanceOutcome::FemMaterialRejected{..}=>unreachable!("C ABI cannot return a FEM retry"),
                        crate::driver::HostAdvanceOutcome::SwingSampleRejected{..}=>unreachable!("C ABI cannot return a swing retry"),
                        crate::driver::HostAdvanceOutcome::ChartPathRejected{..}=>unreachable!("C ABI cannot return a host chart retry"),
                         crate::driver::HostAdvanceOutcome::MaterialSpinRejected{..}=>unreachable!("C ABI cannot return a material spin retry"),
                         crate::driver::HostAdvanceOutcome::NewtonCapRejected{..}=>unreachable!("C ABI cannot return a Newton retry"),
                         crate::driver::HostAdvanceOutcome::SaveRequested{..}=>unreachable!("C ABI cannot return a host save request"),
                    },
                    |token| unsafe { adaptive_retry_cancel(token) },
                    move || unsafe { finish_seed.borrow_mut().finish() });
                if adaptive_retry {crate::driver::adaptive_retry::run_adaptive_attempts(&mut retry_ops,selection.initial_trial_h,logical_start_time,crate::driver::adaptive_retry::MAX_REJECTIONS)} else {
                use crate::driver::adaptive_retry::RetryAttemptOps as _;
                let attempt = retry_ops.prepare_and_advance(nominal_h, 0, 0, 0);
                retry_ops.finish();
                crate::driver::adaptive_retry::RetryRun { save_requested:false, result: attempt.result, requested_h: nominal_h, retry_count: 0, events: Vec::new(),cg_ceiling_candidate:None,final_remaining_budget:None }
                }
            };
            // When adaptive CG is disabled, logical_step's legacy no-host
            // branch does not install a dt restore guard. A swing-only host
            // refusal may have changed dt even though the authored dt did not.
            if (host_swing_retry || host_kinematics_retry) && !adaptive_retry {param.dt=nominal_h;}
            run
                },
            );
            if retry_run.save_requested {retry_save_latched=true;continue;}
            let result = retry_run.result;
            if !result.success() {
                // Derive the crash sub-kind from the StepResult booleans the
                // backend already returns (no log parsing).
                let kind = crash_kind_from_step(
                    result.ccd_success,
                    result.pcg_success,
                    result.intersection_free,
                    result.newton_progress,
                    result.pin_feasible,
                    result.contact_separated,
                );
                let detail = format!(
                    "advance failed at frame {} (ccd={}, pcg={}, \
                     intersection_free={}, newton_progress={}, \
                     pin_feasible={}, contact_separated={})",
                    self.state.curr_frame,
                    result.ccd_success,
                    result.pcg_success,
                    result.intersection_free,
                    result.newton_progress,
                    result.pin_feasible,
                    result.contact_separated
                );
                // The terminal record is the crash report proper, and it is
                // stamped FIRST so nothing written afterward can cost it. It is
                // durable (tmp + fsync + rename) and first-writer-wins, so once
                // it lands the panic hook sees a terminal outcome present and
                // skips its Crashed{Panic}: a later failure while writing the
                // diagnostics can no longer downgrade a precise sub-kind to a
                // generic panic.
                if let Some(owned)=joint_owned.as_ref() { crate::native_joint::write_receipt(&program_args.output,owned,joint_accepted,self.state.time,"STEP_REJECTED",filtered); }
                if let Some(owned)=soft_owned.as_ref() { crate::soft_joint::write_receipt(std::path::Path::new(&program_args.output),owned,joint_accepted,self.state.time,"STEP_REJECTED").expect("FEM joint receipt"); }
                if let Some(p)=frame_publisher.as_mut() { p.barrier().expect("drain frames before rejected-step status"); }
                crate::status_writer::terminal_crash(kind, detail.clone());

                // The diagnostics are written after the terminal record, so a
                // failure while writing them cannot cost the crash report
                // itself.
                let records = fetch_records();
                write_intersection_records(
                    &program_args.output,
                    &records,
                    if sim_args.world_scaling > 0.0 {
                        1.0 / sim_args.world_scaling
                    } else {
                        1.0
                    },
                );
                panic!("failed to advance");
            }
            // Step start/end times bracket this advance. `t_prev` is the time
            // before the step, `t_curr` the (possibly overshot) time after.
            // Both are handed to write_frame_outputs so each crossed frame is
            // interpolated onto its exact frame time rather than snapped to the
            // overshot post-step time (which drifts as a sawtooth in relative
            // frame time when the substep dt does not divide 1/fps).
            let t_prev = self.state.time;
            self.state.time = result.time;
            self.checkpoint_adaptive=Some(adaptive_controller.clone());
            self.checkpoint_adaptive_start=logical_start_time;
            if let Some(owned)=joint_owned.as_ref() {
                let _receipt_phase = crate::driver::phase::start("solver.host_native_step_receipt");
                joint_accepted+=1;
                self.checkpoint_accepted=joint_accepted;
                crate::native_joint::write_receipt(&program_args.output,owned,joint_accepted,self.state.time,"RUNNING",filtered);
            }
            if let Some(owned)=soft_owned.as_ref() {
                let _receipt_phase = crate::driver::phase::start("solver.host_fem_step_receipt");
                crate::soft_joint::write_receipt(std::path::Path::new(&program_args.output),owned,joint_accepted,self.state.time,"RUNNING").expect("FEM joint receipt");
            }
            let t_curr = self.state.time;

            // Frame detection now happens AFTER advance: the post-step
            // state is what gets recorded, with pin and free verts
            // both at sim time = self.state.time.
            let new_frame = (self.state.time * sim_args.fps).floor() as i32;
            if new_frame > self.state.curr_frame {
                // A single advance can cross more than one frame boundary when
                // the timestep spans several frame periods (dt * fps >= 2).
                // Emit every integer frame in the crossed span so the output
                // stays contiguous (downstream readers index frames
                // sequentially); each crossed index is interpolated to its own
                // exact frame time t_frame = f / fps within [t_prev, t_curr].
                for f in self.state.curr_frame + 1..=new_frame.min(requested_frames) {
                    let t_frame = f as f64 / sim_args.fps;
let _wphase = crate::driver::phase::start("solver.write_frame");
                    self.write_frame_outputs(
                        program_args,
                        sim_args,
                        &dataset,
                        &param,
                        &scene,
                        &mut last_time,
                        f,
                        t_frame,
                        t_prev,
                        t_curr,
                        &mut timeline_statistics,
                        joint_accepted,
                        &mut frame_publisher,
                        f == new_frame,
                    );
drop(_wphase);
                }
                self.state.curr_frame = new_frame;
                // A FIELD GRID IS ZERO OUTSIDE ITS BOX, so a cloth blown out of
                // the domain stops being pushed. Said once per frame while it
                // is true, so the log shows it rather than the motion hiding it.
                if let Some(outside) = crate::driver::force_field_outside_count() {
                    if outside > 0 {
                        info!(
                            "force field: {outside} free vertices are outside every grid at frame {new_frame}"
                        );
                    }
                }
                {
                    let _status_phase = crate::driver::phase::start("solver.host_status_progress");
                    if frame_publisher.is_none() { crate::status_writer::progress(Phase::Running, new_frame, self.state.time); }
                }
                // Fire the crash injection once on the final frame, preserving
                // the existing per-advance crash semantics.
                if self.state.curr_frame == sim_args.fake_crash_frame {
                    if let Some(p)=frame_publisher.as_mut() { p.barrier().expect("drain frames before fake crash"); }
                    panic!("fake crash!");
                }
            }
        }
        if let Some(p)=frame_publisher.as_mut() { p.finish().expect("final frame publisher drain"); }
        if restoring.is_some() {
            let runtime=crate::driver::restoring_runtime_receipt().unwrap_or_else(|fatal|panic!("restoring receipt: {}",fatal.detail));
            let backend_gate=crate::driver::restoring_loaded_gate().unwrap_or_else(|fatal|panic!("restoring loaded gate: {}",fatal.detail));
            let seal=restoring_seal;
            let receipt=serde_json::json!({"kind":if seal.is_some(){"ppf.restoring_curves.native_receipt"}else{"ppf.cross_links.native_receipt"},"version":1,"status":"FINISHED","physical_time":self.state.time,
                "solver_identity":crate::compiled_restoring_identity(),
                "backend_gate":backend_gate,"basis_sha256":seal.map(|s|&s.sidecar.basis_sha256),"sidecar_sha256":seal.map(|s|&s.sidecar_sha256),"combined_inputs_sha256":seal.map(|s|&s.combined_inputs_sha256),
                "sealed_mass_verification":if seal.is_some(){serde_json::json!({"status":"PASS","compared":"final builder target mapping, sealed rowhash/stitch epsilon/rest, receipt metrics/quality/eigenvalues/anisotropy, independently recomputed SI K/C","audit_authority_hash_compared":false})}else{serde_json::Value::Null},
                "internal_cross_links":scene.joint_input.as_ref().map(|i|crate::cross_links::receipt_identity(i)),
                "runtime":runtime});
            std::fs::write(std::path::Path::new(&program_args.output).join(if seal.is_some(){"restoring_curves_native_receipt.json"}else{"cross_links_native_receipt.json"}),serde_json::to_vec_pretty(&receipt).expect("restoring receipt serialize")).expect("restoring receipt write");
        }
        write_current_time_to_file(finished_path.to_str().unwrap()).unwrap();
    }
}

/// The hold a caller asked for, `(frame, timeout)`, or `None` when there is
/// none. A file that does not parse stops the run: a caller that wrote a hold
/// and got a run that ignored it would lose exactly the frames it wanted.
fn read_hold(output: &str) -> Option<(i32, std::time::Duration)> {
    let path = std::path::Path::new(output).join(ppf_cts_formats::files::HOLD_AT_FRAME);
    let text = std::fs::read_to_string(&path).ok()?;
    let mut words = text.split_whitespace();
    let frame = words.next().and_then(|w| w.parse::<i32>().ok());
    let timeout = match words.next() {
        None => Some(3600.0),
        Some(w) => w.parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0),
    };
    match (frame, timeout) {
        (Some(frame), Some(timeout)) => {
            Some((frame, std::time::Duration::from_secs_f64(timeout)))
        }
        // An empty file is a caller mid-write; the next check reads it whole.
        _ if text.trim().is_empty() => None,
        _ => panic!(
            "{}: {:?} is not `<frame> [<timeout seconds>]`",
            path.display(),
            text
        ),
    }
}

/// Interpolation weight that places an output frame at its exact frame time.
///
/// `t_prev` is the sim time whose pose lives in `prev_vertex` (step start) and
/// `t_curr` the time in `curr_vertex` (step end, which overshoots the frame
/// boundary because the substep dt rarely divides `1/fps`). `t_frame` is the
/// exact boundary `frame / fps`. Returns the lerp weight `alpha` in `[0, 1]`
/// such that `prev + alpha * (curr - prev)` is the pose at `t_frame`. A
/// zero/degenerate span (frame 0 rest pose where `t_prev == t_curr`) returns
/// `1.0`, which writes `curr_vertex` verbatim.
fn frame_interp_alpha(t_frame: f64, t_prev: f64, t_curr: f64) -> f32 {
    let span = t_curr - t_prev;
    if span > 1e-12 {
        (((t_frame - t_prev) / span).clamp(0.0, 1.0)) as f32
    } else {
        1.0
    }
}

/// Resolve the world-space spin axis for an angular velocity override from a
/// body's live positions. `positions` is packed `(x, y, z)` per vertex
/// (length `3 * count`) as gathered from the GPU at the current step.
/// Returns the centroid and the unit principal axis `pca_index`
/// (0 = largest extent ... 2 = thinnest), eigenvalues sorted descending to
/// match the `pdrd_hinge_axis` convention. The eigenvector sign is
/// canonicalized (the component of largest magnitude is made positive) so a
/// signed angular speed has a well-defined handedness that agrees with the
/// Blender overlay. Returns `None` when the body is too small or the chosen
/// axis is degenerate.
/// Mean of the gathered positions (packed `(x, y, z)` per vertex). Used as
/// the pivot for a fixed world-axis angular override. Returns `None` if empty.
fn centroid_of(positions: &[f32]) -> Option<[f32; 3]> {
    let count = positions.len() / 3;
    if count == 0 {
        return None;
    }
    let mut c = [0.0f64; 3];
    for k in 0..count {
        c[0] += positions[3 * k] as f64;
        c[1] += positions[3 * k + 1] as f64;
        c[2] += positions[3 * k + 2] as f64;
    }
    let inv = 1.0 / count as f64;
    Some([
        (c[0] * inv) as f32,
        (c[1] * inv) as f32,
        (c[2] * inv) as f32,
    ])
}

fn resolve_principal_axis(positions: &[f32], pca_index: usize) -> Option<([f32; 3], [f32; 3])> {
    let count = positions.len() / 3;
    if count < 3 {
        return None;
    }
    let mut centroid = na::Vector3::<f64>::zeros();
    for k in 0..count {
        centroid += na::Vector3::new(
            positions[3 * k] as f64,
            positions[3 * k + 1] as f64,
            positions[3 * k + 2] as f64,
        );
    }
    centroid /= count as f64;

    let mut cov = na::Matrix3::<f64>::zeros();
    for k in 0..count {
        let r = na::Vector3::new(
            positions[3 * k] as f64,
            positions[3 * k + 1] as f64,
            positions[3 * k + 2] as f64,
        ) - centroid;
        cov += r * r.transpose();
    }

    let eig = cov.symmetric_eigen();
    // Sort axes by DESCENDING eigenvalue (0 = largest extent, 2 = thinnest),
    // matching the hinge / Blender PCA convention.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut axis = eig.eigenvectors.column(order[pca_index]).into_owned();
    let norm = axis.norm();
    if !norm.is_finite() || norm <= 1e-12 {
        return None;
    }
    axis /= norm;

    // Canonicalize sign: component of largest magnitude is made positive.
    let mut jmax = 0usize;
    for j in 1..3 {
        if axis[j].abs() > axis[jmax].abs() {
            jmax = j;
        }
    }
    if axis[jmax] < 0.0 {
        axis = -axis;
    }

    Some((
        [centroid[0] as f32, centroid[1] as f32, centroid[2] as f32],
        [axis[0] as f32, axis[1] as f32, axis[2] as f32],
    ))
}

fn write_current_time_to_file(file_path: &str) -> std::io::Result<()> {
    let now = Local::now();
    let time_str = now.to_rfc3339();
    let path = Path::new(file_path);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{time_str}")?;
    Ok(())
}

/// Pull the intersection records the backend latched during the failed step.
///
/// Fetched once here and handed on, rather than fetched again inside each
/// writer: two independent fetches could report different record sets, and a
/// reader comparing the resulting artifacts would have no way to tell which was
/// which.
///
/// The buffer length and the max_count handed across the FFI both use the Rust
/// mirror of the capacity, which must equal the canonical
/// MAX_INTERSECTION_RECORDS macro in `src/kernels/data_records.hpp` that bounds
/// the C++-side writes; if they diverge the C++ getter clamps to its own macro
/// and silently under-reports.
fn fetch_records() -> Vec<IntersectionRecord> {
    let mut records = vec![IntersectionRecord::default(); MAX_INTERSECTION_RECORDS];
    let count = unsafe {
        fetch_intersection_records(records.as_mut_ptr(), MAX_INTERSECTION_RECORDS as u32)
    };
    records.truncate(count as usize);
    records
}

fn write_intersection_records(output_dir: &str, records: &[IntersectionRecord], inv_world: f32) {
    if records.is_empty() {
        return;
    }
    let count = records.len();
    let type_name = |t: u32| match t {
        0 => "face_edge",
        1 => "edge_edge",
        2 => "collision_mesh",
        3 => "point_point",
        _ => "unknown",
    };
    let json_records: Vec<serde_json::Value> = records
        .iter()
        .map(|r| {
            let n0 = r.num_verts0 as usize;
            let n1 = r.num_verts1 as usize;
            // Positions come back at sim scale; divide by world_scaling (via
            // inv_world) so the overlay markers land at the authored scale.
            let mut pos0 = Vec::new();
            for i in 0..n0 {
                pos0.push(vec![
                    r.positions[i * 3] * inv_world,
                    r.positions[i * 3 + 1] * inv_world,
                    r.positions[i * 3 + 2] * inv_world,
                ]);
            }
            let mut pos1 = Vec::new();
            for i in 0..n1 {
                let base = n0 * 3 + i * 3;
                pos1.push(vec![
                    r.positions[base] * inv_world,
                    r.positions[base + 1] * inv_world,
                    r.positions[base + 2] * inv_world,
                ]);
            }
            serde_json::json!({
                "type": type_name(r.itype),
                "elem0": r.elem0,
                "elem1": r.elem1,
                "positions0": pos0,
                "positions1": pos1,
            })
        })
        .collect();
    let json = serde_json::json!({
        "count": count,
        "records": json_records,
    });
    // Write via a temp file + atomic rename so a crash mid-write never
    // leaves a half-written records file for the overlay / server to read.
    // Every failure names its path and the OS error: on the crash path, a
    // message carrying neither leaves nothing to act on.
    let path = format!(
        "{output_dir}/{}",
        ppf_cts_formats::files::INTERSECTION_RECORDS_JSON
    );
    let tmp = format!("{path}.tmp");
    let write_result = std::fs::File::create(&tmp)
        .map_err(|e| format!("create {tmp}: {e}"))
        .and_then(|file| {
            serde_json::to_writer_pretty(file, &json).map_err(|e| format!("write {tmp}: {e}"))
        })
        .and_then(|()| std::fs::rename(&tmp, &path).map_err(|e| format!("rename to {path}: {e}")));
    match write_result {
        Ok(()) => info!("wrote {count} intersection records to {path}"),
        Err(msg) => {
            error!("failed to write intersection records: {msg}");
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

#[cfg(test)]
mod rest_shape_tests {
    use super::*;

    // The time-varying rest-shape keyframe interpolation used per step.
    // Endpoints must reproduce the keyframes exactly (alpha 0), the interior
    // is a component-wise lerp, and out-of-range times clamp to the ends.
    #[test]
    fn interp_rest_shape_endpoints_midpoint_and_clamp() {
        let i2 = Mat2x2f::identity();
        let i3 = Mat3x3f::identity();
        let keyframes: Vec<RestShapeKeyframe> = vec![
            (0.0_f64, vec![i2], vec![i3], vec![0u8], vec![0u8]),
            (
                1.0_f64,
                vec![i2 * 2.0],
                vec![i3 * 2.0],
                vec![0u8],
                vec![0u8],
            ),
        ];

        let r0 = interp_rest_shape(&keyframes, 0.0);
        assert!((r0.inv_rest2x2.as_slice()[0] - i2).amax() < 1e-6);
        assert!((r0.inv_rest3x3.as_slice()[0] - i3).amax() < 1e-6);

        let r1 = interp_rest_shape(&keyframes, 1.0);
        assert!((r1.inv_rest2x2.as_slice()[0] - i2 * 2.0).amax() < 1e-6);
        assert!((r1.inv_rest3x3.as_slice()[0] - i3 * 2.0).amax() < 1e-6);

        let rm = interp_rest_shape(&keyframes, 0.5);
        assert!((rm.inv_rest2x2.as_slice()[0] - i2 * 1.5).amax() < 1e-6);
        assert!((rm.inv_rest3x3.as_slice()[0] - i3 * 1.5).amax() < 1e-6);

        // Below the first keyframe clamps to frame 0.
        let rb = interp_rest_shape(&keyframes, -5.0);
        assert!((rb.inv_rest2x2.as_slice()[0] - i2).amax() < 1e-6);
        // Above the last keyframe clamps to the final frame.
        let ra = interp_rest_shape(&keyframes, 5.0);
        assert!((ra.inv_rest2x2.as_slice()[0] - i2 * 2.0).amax() < 1e-6);
    }

    // The exclude mask is the OR of the two bracketing keyframes, so an element
    // near-singular in either adjacent sample is dropped for the whole interval.
    #[test]
    fn interp_rest_shape_excludes_union_of_bracketing() {
        let i3 = Mat3x3f::identity();
        // two tets: tet0 singular only at the second keyframe, tet1 never.
        let keyframes: Vec<RestShapeKeyframe> = vec![
            (0.0_f64, vec![], vec![i3, i3], vec![], vec![0u8, 0u8]),
            (1.0_f64, vec![], vec![i3, i3], vec![], vec![1u8, 0u8]),
        ];
        // Mid-interval: tet0 excluded (OR), tet1 not.
        let rm = interp_rest_shape(&keyframes, 0.5);
        assert_eq!(rm.exclude_tet.as_slice(), &[1u8, 0u8]);
        // At the first keyframe exactly, still excluded for tet0 (i0==i1==1? no:
        // time 0.0 -> i0=i1=0, mask = 0|0). Confirm the clamp picks frame 0.
        let r0 = interp_rest_shape(&keyframes, 0.0);
        assert_eq!(r0.exclude_tet.as_slice(), &[0u8, 0u8]);
    }

    // A single-keyframe schedule must not panic and returns that frame for any
    // query time (the degenerate static-rest case).
    #[test]
    fn interp_rest_shape_single_keyframe() {
        let i3 = Mat3x3f::identity();
        let keyframes: Vec<RestShapeKeyframe> = vec![(
            0.5_f64,
            Vec::<Mat2x2f>::new(),
            vec![i3 * 3.0],
            vec![],
            vec![0u8],
        )];
        let r = interp_rest_shape(&keyframes, 2.0);
        assert!(r.inv_rest2x2.as_slice().is_empty());
        assert!((r.inv_rest3x3.as_slice()[0] - i3 * 3.0).amax() < 1e-6);
    }
}

#[cfg(test)]
mod frame_interp_tests {
    use super::frame_interp_alpha;

    // dt=0.01, fps=30 (3.333 substeps/frame): the step that crosses frame 1
    // runs from t=0.03 to 0.04, and the exact boundary 1/30 = 0.03333 sits one
    // third of the way in. Without interpolation the frame would be recorded at
    // the overshot 0.04 (a 0.2-frame error); the weight pulls it back to 0.0333.
    #[test]
    fn alpha_places_frame_on_the_exact_boundary() {
        let a = frame_interp_alpha(1.0 / 30.0, 0.03, 0.04);
        assert!((a - 1.0 / 3.0).abs() < 1e-5, "alpha={a}");
    }

    // A full step that lands exactly on the boundary (alpha 1) writes curr; a
    // boundary sitting exactly on the step start writes prev (alpha 0). These
    // are the integer-substeps-per-frame endpoints, where the output is exact.
    #[test]
    fn alpha_endpoints_are_exact() {
        assert_eq!(frame_interp_alpha(0.04, 0.03, 0.04), 1.0);
        assert_eq!(frame_interp_alpha(0.03, 0.03, 0.04), 0.0);
    }

    // The rest pose (frame 0) is emitted with t_prev == t_curr; the degenerate
    // span must collapse to curr (weight 1.0) instead of dividing by zero.
    #[test]
    fn alpha_zero_span_is_curr() {
        assert_eq!(frame_interp_alpha(0.0, 0.0, 0.0), 1.0);
    }

    // A boundary outside the bracketing step (should never happen given the
    // caller only emits crossed frames) clamps into range rather than
    // extrapolating past prev/curr.
    #[test]
    fn alpha_clamps_out_of_range() {
        assert_eq!(frame_interp_alpha(0.05, 0.03, 0.04), 1.0);
        assert_eq!(frame_interp_alpha(0.02, 0.03, 0.04), 0.0);
    }
}

#[cfg(test)]
mod collision_checkpoint_save_tests {
    use super::*;
    use std::{fs,path::PathBuf,panic::{catch_unwind,AssertUnwindSafe}};
    #[derive(Deserialize)] struct Config { param:SimArgs }
    #[test]
    #[ignore="requires the actual no-mask CPU checkpoint fixture"]
    fn collision_filter_existing_dataset_save_refuses_uncertified_bytes_before_marker_invalidation() {
        let source=PathBuf::from(std::env::var("PPF_COLLISION_CHECKPOINT_ROOT").expect("actual CPU checkpoint root"));
        let area=PathBuf::from(std::env::var("PPF_COLLISION_TEST_AREA").expect("task test area"));
        let config:Config=toml::from_str(&fs::read_to_string(source.join("../checkpoint-no-mask-input/param.toml")).unwrap()).unwrap();
        let mut backend=Backend::load_state(1,source.to_str().unwrap());
        let dataset:DataSet=crate::read(&crate::read_gz(source.join(ppf_cts_formats::files::DATASET).to_str().unwrap()));
        for case in ["missing","old","hash-mismatch","valid"] {
            let dir=area.join(format!("save-{}-{}-{case}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));fs::create_dir_all(&dir).unwrap();
            let path=dir.join(ppf_cts_formats::files::DATASET);fs::copy(source.join(ppf_cts_formats::files::DATASET),&path).unwrap();
            let marker=dir.join(crate::checkpoint_layout::MARKER);
            match case {
                "missing"=>(),
                "old"=>{fs::write(&path,b"legacy positional dataset bytes").unwrap();fs::write(&marker,b"old-layout\n").unwrap();},
                "hash-mismatch"=>{crate::checkpoint_layout::record(&dir);fs::write(&path,b"changed dataset bytes").unwrap();},
                _=>crate::checkpoint_layout::record(&dir),
            }
            let before=fs::read(&path).unwrap();let marker_before=fs::read(&marker).ok();
            let args=ProgramArgs{twist_resume_mode:None,path:source.to_string_lossy().into_owned(),output:dir.to_string_lossy().into_owned(),load:0,require_collision_filter_v1:false,require_pcg_precision_v1:false,require_restoring_curves_v1:false,restoring_curves_mass_query:None,target_frame:None,resume_capability_json:false,async_output:false};
            let mut param=crate::builder::make_param(&config.param);
            param.time=backend.state.time;
            let result=catch_unwind(AssertUnwindSafe(||backend.save_state(&args,&config.param,&dataset,&param)));
            if case=="valid" {
                result.expect("certified existing dataset may be saved again");
                crate::checkpoint_layout::validate(&dir).unwrap();
                assert!(dir.join(ppf_cts_formats::files::state_filename(1)).is_file());
            } else {
                let err=result.expect_err("uncertified existing dataset must refuse before any marker write");
                let detail=err.downcast_ref::<String>().map(String::as_str).or_else(||err.downcast_ref::<&str>().copied()).unwrap_or("");
                assert!(detail.contains("CHECKPOINT_LAYOUT_UNSUPPORTED")&&detail.contains("fresh output directory"),"{detail}");
                assert_eq!(fs::read(&marker).ok(),marker_before,"no new marker may certify old bytes");
                assert_eq!(fs::read_dir(&dir).unwrap().count(),if marker_before.is_some(){2}else{1},"refusal writes no mesh/state/plasticity/temporary file");
            }
            assert_eq!(fs::read(&path).unwrap(),before,"existing dataset bytes remain unchanged");
        }
    }
}



#[cfg(all(test, feature="cpu"))]
#[path="backend_save_fixture.rs"]
mod backend_save_fixture;

