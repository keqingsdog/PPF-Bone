//! Sealed SI evidence is checked against the final builder before any upload.
use crate::{scene::Scene,data::DataSet,restoring_curve as wire,restoring_curve_gpu as gpu};
pub struct Installed {pub tables:wire::PackedTables,pub bindings:Vec<gpu::CurveBinding>}
/// Read the ABI producer actually linked to the running device. Values in this
/// receipt come from the loaded probe, not expected_layout or a configured file.
pub fn loaded_gate(device:&impl ppf_cts_compute::Device)->Result<serde_json::Value,String> {
    #[cfg(abi_backend_linked)] {
        extern "C" {fn be_abi_version()->u32;fn be_layout_probe(out:*mut ppf_cts_compute::abi::BeLayoutProbe);}
        let version=unsafe{be_abi_version()};
        if version!=ppf_cts_compute::abi::ABI_VERSION{return Err("RESTORING_CAPABILITY_UNAVAILABLE: actual ABI changed".into());}
        let mut probe=ppf_cts_compute::abi::BeLayoutProbe::default();unsafe{be_layout_probe(&mut probe)};
        ppf_cts_compute::abi::compare_layout(&probe).map_err(|e|format!("RESTORING_CAPABILITY_UNAVAILABLE: {e}"))?;
        ppf_cts_compute::abi::compare_restoring_layout(&probe).map_err(|e|format!("RESTORING_CAPABILITY_UNAVAILABLE: {e}"))?;
        let hex=|v:[u8;32]|v.iter().map(|b|format!("{b:02x}")).collect::<String>();
        return Ok(serde_json::json!({"source":"ACTUAL_LOADED_BE_LAYOUT_PROBE_ON_RUNNING_DEVICE","backend":device.info().backend,
            "device_name":device.info().device_name,"abi_version":version,"contract_version":probe.restoring_curve_contract_version,
            "channel_mask":probe.restoring_curve_channel_mask,"layout_sha256":hex(probe.restoring_curve_layout_sha256),
            "kernel_table_sha256":hex(probe.restoring_curve_kernel_table_sha256)}));
    }
    #[cfg(not(abi_backend_linked))] {
        Ok(serde_json::json!({"source":"ACTUAL_HOST_DEVICE","backend":device.info().backend,"device_name":device.info().device_name}))
    }
}
pub fn verify_and_pack(prepared:&wire::PreparedCurves,scene:&Scene,data:&DataSet,root:&std::path::Path)->Result<Installed,String> {
    prepared.verify_unchanged(root)?;
    wire::verify_mass(&prepared.sidecar,|target|crate::restoring_mass_authority::recompute(scene,data,&prepared.snapshot,target))?;
    for receipt in &prepared.sidecar.mass_receipts {
        let actual=crate::restoring_mass_authority::recompute(scene,data,&prepared.snapshot,&receipt.target)?;
        let omega=2.*std::f64::consts::PI*receipt.frequency_hz;
        let close=|x:f64,y:f64|x.is_finite()&&y.is_finite()&&(x-y).abs()<=x.abs().max(y.abs())*1e-6;
        if !close(actual.metric_eff*omega*omega,receipt.resolved_stiffness)
            || !close(2.*receipt.damping_ratio*actual.metric_eff*omega,receipt.resolved_damping) {
            return Err("RESTORING_MASS_INVALID: independently recomputed SI K/C".into());
        }
    }
    for b in &prepared.sidecar.bindings {
        let actual=crate::restoring_mass_authority::target_geometry(scene,data,&prepared.snapshot,&b.target)?;
        match (&b.target,&actual) {
            (wire::Target::Stitch{row_index,rows_sha256,epsilon_m,rest_length_m},wire::Target::Stitch{row_index:a,rows_sha256:h,epsilon_m:e,rest_length_m:l})=>{
                let close=|x:f64,y:f64|x.is_finite()&&y.is_finite()&&(x-y).abs()<=y.abs()*1e-6;
                if row_index!=a||rows_sha256!=h||!close(*epsilon_m,*e)||!close(*rest_length_m,*l){return Err("RESTORING_TARGET_INVALID: sealed stitch geometry".into());}
            },(wire::Target::Joint{..},wire::Target::Joint{..})=>if b.target!=actual{return Err("RESTORING_TARGET_INVALID: joint identity".into());},
            _=>return Err("RESTORING_TARGET_INVALID: target family changed".into()),
        }
    }
    let bindings=wire::pack_bindings(&prepared.sidecar,|target|crate::restoring_mass_authority::target_slot(scene,data,&prepared.snapshot,target))?;
    let (tables,bindings)=gpu::solver_space(&prepared.packed,&bindings,scene.args().world_scaling as f64)?;
    prepared.verify_unchanged(root)?;
    Ok(Installed{tables,bindings})
}
