//! Draft: copy beside pcg.rs and enable from pcg.rs after the frozen builds.
use super::*;
use ppf_cts_compute::{AllocLabel,Counters,DeviceInfo,Diag,Encoder,Handle,HostDevice,KernelDecl,KernelId};

struct RejectWarmAlloc { inner: HostDevice, label: &'static str, runs:usize }
impl Device for RejectWarmAlloc {
    type Region = <HostDevice as Device>::Region;
    fn info(&self)->&DeviceInfo {self.inner.info()}
    fn kernels(&self)->&'static [KernelDecl] {self.inner.kernels()}
    fn missing(&self)->&[KernelId] {self.inner.missing()}
    fn prepare_kernels(&mut self, ids:&[KernelId])->Result<(),Fault>{self.inner.prepare_kernels(ids)}
    fn alloc(&mut self,count:usize,elem_size:usize,align:usize,label:AllocLabel)->Result<Handle,Fault>{
        if label.0 == self.label {
            return Err(Fault::Alloc {label,bytes:count.saturating_mul(elem_size),detail:"simulated backend device failure".into()});
        }
        self.inner.alloc(count,elem_size,align,label)
    }
    fn grow(&mut self,h:&mut Handle,n:usize,e:usize,a:usize)->Result<(),Fault>{self.inner.grow(h,n,e,a)}
    fn free(&mut self,h:&mut Handle)->Result<(),Fault>{self.inner.free(h)}
    fn write(&mut self,h:Handle,o:usize,src:&[u8])->Result<(),Fault>{self.inner.write(h,o,src)}
    fn read(&mut self,h:Handle,o:usize,dst:&mut [u8])->Result<(),Fault>{self.inner.read(h,o,dst)}
    fn copy(&mut self,d:Handle,doff:usize,s:Handle,soff:usize,n:usize)->Result<(),Fault>{self.inner.copy(d,doff,s,soff,n)}
    fn allocator_generation(&self)->u64{self.inner.allocator_generation()}
    fn bytes_reserved(&self)->u64{self.inner.bytes_reserved()}
    fn run<F>(&mut self,region:&'static str,f:F)->Result<Diag,Fault> where F:FnOnce(&mut dyn Encoder)->Result<(),Fault>{self.runs+=1;self.inner.run(region,f)}
    fn record<F>(&mut self,region:&'static str,f:F)->Result<Self::Region,Fault> where F:FnOnce(&mut dyn Encoder)->Result<(),Fault>{self.inner.record(region,f)}
    fn replay(&mut self,region:&Self::Region,n:u32)->Result<Diag,Fault>{self.inner.replay(region,n)}
    fn release(&mut self,region:Self::Region){self.inner.release(region)}
    fn counters(&self)->Counters{self.inner.counters()}
    fn counters_reset(&mut self){self.inner.counters_reset()}
}

#[test]
fn warm_scratch_preserves_candidate_and_cold_seed_across_batch_style_updates() {
    fn buffer(d:&mut impl Device, values:&[f32])->Buffer<f32> {
        let mut b=Buffer::default();b.size(d,values.len(),AllocLabel("test.warm.alias")).unwrap();
        b.write(d,0,values).unwrap();b
    }
    let mut d=super::super::launch::host_device();
    let current=buffer(&mut d,&[5.,4.,-2.]);let cold=buffer(&mut d,&[1.,-1.,3.]);
    let mut first=buffer(&mut d,&[0.;3]);let history=buffer(&mut d,&[2.,-4.,8.]);
    let mut candidate=buffer(&mut d,&[0.;3]);
    warm_difference(&mut d,current.handle(),cold.handle(),first.handle(),3).unwrap();
    warm_add_second(&mut d,first.handle(),history.handle(),candidate.handle(),0.5,3).unwrap();
    let mut result=[0.;3];candidate.read(&mut d,0,&mut result).unwrap();assert_eq!(result,[5.,3.,-1.]);
    // Simulate a later iterate update, then publish the homogeneous correction
    // against the original cold seed, as on a verified batch finish.
    let after=buffer(&mut d,&[8.,1.,-2.]);
    warm_difference(&mut d,after.handle(),cold.handle(),first.handle(),3).unwrap();
    first.read(&mut d,0,&mut result).unwrap();assert_eq!(result,[7.,2.,-5.]);
    let mut seed=[0.;3];cold.read(&mut d,0,&mut seed).unwrap();assert_eq!(seed,[1.,-1.,3.]);
}

#[test]
fn backend_alloc_fault_invalidates_history_and_never_reaches_cold_solve() {
    const CHILD_ENV: &str = "PPF_WARM_ALLOC_FAULT_TEST_CHILD";
    const MARKER: &str = "PPF_WARM_ALLOC_FAULT_TEST_BODY_ENTERED";
    if std::env::var(CHILD_ENV).as_deref() != Ok("1") {
        // module_path! includes the crate name; libtest's exact name does not.
        let module = module_path!().split_once("::").expect("test module path").1;
        let test_name = format!("{module}::backend_alloc_fault_invalidates_history_and_never_reaches_cold_solve");
        let output = std::process::Command::new(std::env::current_exe().expect("current test binary"))
            .args(["--exact", &test_name, "--nocapture"])
            .env(CHILD_ENV, "1")
            .env("PPF_PCG_WARM_START", "2")
            .output()
            .expect("run isolated warm-allocation test");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "isolated test failed:\n{stdout}\n{stderr}");
        assert!(stdout.contains("running 1 test") && stderr.contains(MARKER),
            "isolated test did not execute its body:\n{stdout}\n{stderr}");
        return;
    }
    eprintln!("{MARKER}");
    assert_eq!(std::env::var("PPF_PCG_WARM_START").as_deref(),Ok("2"),"isolated child requires PPF_PCG_WARM_START=2");
    for failed in ["pcg.warm.history", "pcg.warm.seed", "pcg.warm.history2", "pcg.warm.ad1", "pcg.warm.ad2"] {
        let mut device=RejectWarmAlloc {inner:super::super::launch::host_device(),label:failed,runs:0};
        let mut workspace=RigidWorkspace::default();
        workspace.warm_key=Some(WarmKey {nrow:1,n_cloth:1,bodies:0,dim:3,body_base:3,
            vbody:vec![u32::MAX],cloth_off:vec![0],mask:vec![0],dt_bits:0.01f32.to_bits(),mode:2});
        workspace.warm_count=2;
        workspace.warm_cold_cost.record_success(40,true,false);
        let map=super::super::rigid_map::RigidMap {nrow:1,n_cloth:1,dim:3,body_base:3,
            vbody:vec![u32::MAX],cloth_off:vec![0],..Default::default()};
        let mut staged=super::super::rigid_map::Staged::default();
        let mut reduction=super::super::pdrd::Reduction{map:&map,staged:&mut staged};
        let fixed=super::super::spmv::FixedCsrView {index:Handle::NONE,offset:Handle::NONE,
            value:Handle::NONE,transpose_pair:Handle::NONE,transpose_offset:Handle::NONE,rows:1};
        let op=super::super::operator::Operator{fixed_pattern: None, tet_factor:None,dynamic:None,fixed,diagonal:Handle::NONE};
        let mut joints=super::super::joints::Joints::default();
        let view=joints.view(Handle::NONE,&map,0.01);
        let mut work=Workspace::default();
        // All other handles deliberately absent: the allocation fault must
        // abort before the first cold operator or kernel can consume them.
        let result=unsafe {solve_rigid_impl(&mut device,&op,&mut reduction,Handle::NONE,Handle::NONE,
            Handle::NONE,Handle::NONE,&mut work,&mut workspace,None,None,view,None,Handle::NONE,
            1e-3,100,RigidSolveInput::NewtonForce,false,true,Some(Handle::NONE),None,FullBodyAdmission::Unqualified)};
        let error=result.expect_err("backend alloc fault must abort solve");
        assert!(error.detail.contains("simulated backend device failure"),"unexpected fatal: {}",error.detail);
        assert!(error.detail.contains(failed),"missing allocation label: {}",error.detail);
        assert!(workspace.warm_key.is_none());
        assert_eq!(workspace.warm_count,0);
        assert_eq!(workspace.warm_cold_cost.samples().0,0);
        assert_eq!(device.runs,0,"cold kernel path must remain unreachable");
    }
}
