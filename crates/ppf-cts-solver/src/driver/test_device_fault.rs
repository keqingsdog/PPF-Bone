//! Private, one-shot CPU test faults at real device-operation result boundaries.
use std::sync::atomic::{AtomicUsize, Ordering};
use ppf_cts_compute::Fault;

#[derive(Clone, Copy)]
pub(crate) enum Point { SeedCaptureCopy, SeedRestoreCopy, SeedCaptureCleanupFree, SeedDiscardFree, ConstraintFixWrite }
static ARMED: AtomicUsize = AtomicUsize::new(0);
static HITS: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
impl Point { fn index(self) -> usize { self as usize } }

pub(crate) struct Armed;
impl Armed {
    pub(crate) fn new(points: &[Point]) -> Self {
        ARMED.store(0, Ordering::SeqCst);
        for hit in &HITS { hit.store(0, Ordering::SeqCst); }
        let bits = points.iter().fold(0, |bits, p| bits | (1 << p.index()));
        ARMED.store(bits, Ordering::SeqCst);
        Self
    }
    pub(crate) fn hits(point: Point) -> usize { HITS[point.index()].load(Ordering::SeqCst) }
}
impl Drop for Armed { fn drop(&mut self) { ARMED.store(0, Ordering::SeqCst); } }
pub(crate) fn check(point: Point) -> Result<(), Fault> {
    let bit = 1 << point.index();
    if ARMED.fetch_and(!bit, Ordering::SeqCst) & bit != 0 {
        HITS[point.index()].fetch_add(1, Ordering::SeqCst);
        return Err(Fault::Platform { call: "private_test_device_operation", detail: format!("injected point {}", point.index()) });
    }
    Ok(())
}

pub(crate) fn start_private_status() {
    let root=std::env::var_os("PPF_PRIVATE_TEST_ROOT").expect("private test output root");
    let nanos=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
    let pid=std::process::id() as u64;
    let id=pid.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(nanos) & 0xFFFF_FFFF_FFFF;
    let dir=std::path::PathBuf::from(root).join(format!("status_{pid}_{nanos}"));
    std::fs::create_dir_all(&dir).expect("private status dir");
    let lock = ppf_cts_formats::status::lock::acquire(&dir).expect("status lock");
    crate::status_writer::init(dir.to_str().expect("status path"),format!("{id:012x}"),false,lock);
}
