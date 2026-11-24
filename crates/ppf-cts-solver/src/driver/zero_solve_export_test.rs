//! End-to-end regression of the real rigid PCG zero-residual return.
use super::*;
use ppf_cts_compute::{AllocLabel, Handle, Pod};
use crate::driver::{launch::host_device, rigid_map::{RigidMap, Staged, RIGID_UNSET}, pdrd::Reduction, operator::Operator, spmv::FixedCsrView, roots::{View as RootView, ConstraintDomain}};

fn buffer<T: Pod>(device: &mut impl Device, values: &[T]) -> Buffer<T> {
    let mut b = Buffer::default();
    b.size(device, values.len(), AllocLabel("test.zero_solve.input")).unwrap();
    b.write(device, 0, values).unwrap();
    b
}

fn run_case(root_enabled: bool, diagnostic: Option<&str>) {
    let mut device = host_device();
    let map = RigidMap { nrow: 4, n_bodies: 1, n_cloth: 0, dim: 6, body_base: 0,
        vbody: vec![0; 4], cloth_off: vec![RIGID_UNSET; 4], ..Default::default() };
    let mut staged = Staged::default();
    staged.vbody.size(&mut device, 4, AllocLabel("test.zero.vbody")).unwrap();
    staged.vbody.at().copy_from_slice(&map.vbody);
    staged.vbody.upload(&mut device).unwrap();
    staged.cloth_off.size(&mut device, 4, AllocLabel("test.zero.cloth")).unwrap();
    staged.cloth_off.at().copy_from_slice(&map.cloth_off);
    staged.cloth_off.upload(&mut device).unwrap();
    let rest = [1.,1.,1., 1.,-1.,-1., -1.,1.,-1., -1.,-1.,1.];
    staged.prot = buffer(&mut device, &rest);
    let offsets = buffer(&mut device, &[0u32; 5]);
    // Valid backing handles even though the fixed CSR has no entries.
    let indices = buffer(&mut device, &[0u32; 1]);
    let values = buffer(&mut device, &[0.0f32; 9]);
    let mut diagonal = [0.0f32; 36];
    for i in 0..4 { for j in 0..3 { diagonal[9*i+4*j] = 1.0; } }
    let diagonal = buffer(&mut device, &diagonal);
    let fixed = FixedCsrView { index: indices.handle(), offset: offsets.handle(), value: values.handle(),
        transpose_pair: indices.handle(), transpose_offset: offsets.handle(), rows: 4 };
    let op = Operator { fixed_pattern: None, tet_factor: None, dynamic: None, fixed, diagonal: diagonal.handle() };
    let b = buffer(&mut device, &[0.0f32; 12]);
    let mut x = buffer(&mut device, &[0.0f32; 12]);
    let mut rot = buffer(&mut device, &[99.0f32; 3]);
    let mut work = Workspace::default(); work.size_for(&mut device, 4).unwrap();
    let mut rigid = RigidWorkspace::default(); rigid.allocate(&mut device, 6, 4, 1).unwrap();
    let state = crate::data::PdrdRigidState { rotation: [1.,0.,0.,0.,1.,0.,0.,0.,1.], count: 4, ..Default::default() };
    let states = buffer(&mut device, &[state]);
    let q = std::f64::consts::FRAC_PI_4;
    let target = crate::joint_root_motion_gpu::RootTargetGpu::sampled(0,
        crate::joint_root_motion_wire::RootPose { center_world_m: [0.;3], rotation_world_wxyz: [q.cos(),0.,0.,q.sin()] }, 0, 4).unwrap();
    let targets = buffer(&mut device, &[target]);
    let root = root_enabled.then_some(RootView { targets: targets.handle(), count: 1, bodies: 1, base: 0, dim: 6, constraint_domain: ConstraintDomain::default() });
    let mut joints = super::super::joints::Joints::default();
    let view = joints.view(states.handle(), &map, 0.01);
    let mut reduction = Reduction { map: &map, staged: &mut staged };
    let result = unsafe { solve_rigid_impl(&mut device, &op, &mut reduction, Handle::NONE, Handle::NONE,
        b.handle(), x.handle(), &mut work, &mut rigid, Some(rot.handle()), None, view, root, Handle::NONE,
        1e-3, 100, RigidSolveInput::NewtonForce, false, true, None, None, FullBodyAdmission::Unqualified) };
    let mut rotation = [0.0;3]; rot.read(&mut device,0,&mut rotation).unwrap();
    let mut full = [0.0;12]; x.read(&mut device,0,&mut full).unwrap();
    if diagnostic.is_some() {
        let error = result.expect_err("diagnostic must refuse early success");
        let expected = match diagnostic.unwrap() { "legacy" => "legacy rows self check not reached; no commit", "batch" => "factor batch8 diagnostic not reached; no commit", _ => unreachable!() };
        assert!(error.detail.contains(expected), "unexpected diagnostic error: {}", error.detail);
        assert_eq!(rotation,[99.0;3], "diagnostic must not publish rotation");
        assert_eq!(full,[0.0;12], "diagnostic must not publish position");
    } else {
        let report = result.expect("real zero residual solve");
        assert!(matches!(report.outcome, Outcome::Converged));
        assert_eq!(report.iterations, 1); // production reports its initial evaluation as iteration 1
        assert_eq!(report.relative_residual, 0.0);
        if root_enabled {
            assert!((rotation[2] + std::f32::consts::FRAC_PI_2).abs() < 1e-5, "root spin: {rotation:?}");
            assert!(rotation[0].abs() < 1e-6 && rotation[1].abs() < 1e-6);
            assert!(full.iter().any(|v|v.abs() > 0.1), "root seed must be prolonged");
        } else {
            assert_eq!(rotation,[0.0;3], "stale rotation must be replaced");
            assert_eq!(full,[0.0;12]);
        }
    }
}

#[test]
fn zero_residual_real_solve_exports_seed_and_diagnostics_do_not_commit() {
    const CHILD: &str = "PPF_ZERO_SOLVE_REGRESSION_CHILD";
    if let Ok(mode) = std::env::var(CHILD) {
        match mode.as_str() {
            "plain" => run_case(false,None),
            "root" => run_case(true,None),
            "legacy" => run_case(true,Some("legacy")),
            "batch" => run_case(true,Some("batch")),
            _ => panic!("unexpected child mode"),
        }
        return;
    }
    let module = module_path!().split_once("::").unwrap().1;
    let name = format!("{module}::zero_residual_real_solve_exports_seed_and_diagnostics_do_not_commit");
    for mode in ["plain", "root", "legacy", "batch"] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", &name, "--nocapture"]).env(CHILD,mode)
            .env("PPF_LEGACY_ROWS_SELF_CHECK","0").env("PPF_FACTOR_BATCH8_PACKET_SELF_CHECK","0")
            .env_remove("PPF_PCG_WARM_START").env_remove("PPF_RESTRICT_GATHER");
        if mode == "legacy" { command.env("PPF_LEGACY_ROWS_SELF_CHECK","1"); }
        if mode == "batch" { command.env("PPF_FACTOR_BATCH8_PACKET_SELF_CHECK","1"); }
        let output=command.output().expect("isolated regression process");
        assert!(output.status.success(), "{mode}:\n{}\n{}",String::from_utf8_lossy(&output.stdout),String::from_utf8_lossy(&output.stderr));
        assert!(String::from_utf8_lossy(&output.stdout).contains("running 1 test"), "child did not execute");
    }
}


