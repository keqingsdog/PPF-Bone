//! Resident fixed CSR body inverse. Caller must explicitly free on the owning device.
use ppf_cts_compute::{AllocLabel, Buffer, Device, Fault, Handle, Pod};
use super::body_inverse_pack::Packed;
use super::spmv::{fixed_csr_spmv, FixedCsrView};

pub struct BodyInverseDevice {
    index: Buffer<u32>, offset: Buffer<u32>, value: Buffer<f32>,
    transpose_pair: Buffer<u32>, transpose_offset: Buffer<u32>, rows: u32, live: bool,
}
fn upload<T: Pod>(device: &mut impl Device, dst: &mut Buffer<T>, data: &[T], label: &'static str) -> Result<(), Fault> {
    dst.size(device, data.len().max(1), AllocLabel(label))?;
    if !data.is_empty() { dst.write(device, 0, data)?; }
    Ok(())
}
impl BodyInverseDevice {
    pub fn upload(device: &mut impl Device, p: &Packed) -> Result<Self, Fault> {
        let mut d = Self { index: Buffer::none(), offset: Buffer::none(), value: Buffer::none(), transpose_pair: Buffer::none(), transpose_offset: Buffer::none(), rows: p.rows, live: true };
        let result = (|| { upload(device, &mut d.index, &p.index, "body_inv.index")?; upload(device, &mut d.offset, &p.offset, "body_inv.offset")?; upload(device, &mut d.value, &p.value, "body_inv.value")?; upload(device, &mut d.transpose_pair, &p.transpose_pair, "body_inv.transpose_pair")?; upload(device, &mut d.transpose_offset, &p.transpose_offset, "body_inv.transpose_offset")?; Ok(()) })();
        if let Err(e) = result { return Err(super::body_factor_device::upload_failure("body inverse upload cleanup", e, d.free(device))); }
        Ok(d)
    }
    /// Call with distinct f32 body spans on the same device. Both allocations and
    /// this object must remain live until the device completes the dispatch.
    pub unsafe fn apply(&self, device: &mut impl Device, x: Handle, out: Handle) -> Result<(), Fault> {
        if !self.live { return Err(Fault::Shape { kernel: "spmv.fixed_csr", detail: "body inverse has been freed".into() }); }
        let count = 3 * self.rows as usize;
        assert_eq!(x.size as usize, count); assert_eq!(out.size as usize, count);
        assert_eq!(x.off % 4, 0); assert_eq!(out.off % 4, 0);
        assert!(!spans_overlap(x, out, count), "input and output overlap");
        let view = FixedCsrView { index:self.index.handle(), offset:self.offset.handle(), value:self.value.handle(), transpose_pair:self.transpose_pair.handle(), transpose_offset:self.transpose_offset.handle(), rows:self.rows };
        fixed_csr_spmv(device, &view, x, out, count)
    }
    pub fn free(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        // Try every release even if an earlier one fails.
        self.live = false;
        let mut result = Ok(());
        for r in [self.index.free(device), self.offset.free(device), self.value.free(device), self.transpose_pair.free(device), self.transpose_offset.free(device)] { if result.is_ok() { result = r; } }
        result
    }
}

/// Opt-in diagnostic, using the actual device dispatcher and an independent dense oracle.
pub fn self_check(device: &mut impl Device) -> Result<(), Fault> {
    let n = 12usize;
    let mut a = vec![0.0f64; n*n];
    for i in 0..n { for j in 0..n { a[i*n+j] = if i==j { 20.0 } else { (i.min(j)*n+i.max(j)+1) as f64 * 0.001 }; } }
    let packed = super::body_inverse_pack::pack(&a, &[2,0], &[1], 3).expect("diagnostic matrix is SPD");
    let mut inv = BodyInverseDevice::upload(device, &packed)?;
    let x: Vec<f32> = (0..18).map(|i| i as f32 - 5.0).collect();
    let mut xd=Buffer::<f32>::none(); let mut yd=Buffer::<f32>::none();
    let result = (|| {
        upload(device, &mut xd, &x, "body_inv.test_x")?;
        yd.size(device, 18, AllocLabel("body_inv.test_y"))?;
        let xh=xd.handle(); let yh=yd.handle();
        assert!(!spans_overlap(xh,yh,18));
        assert!(spans_overlap(xh,xh,18));
        unsafe { inv.apply(device, xh, yh)?; }
        let mut y=vec![0.0f32;18]; yd.read(device,0,&mut y)?;
        for b in 0..3 { for i in 0..6 {
            let mut expected=0.0f64;
            if b!=1 { let fi=if b==2 {0} else {1}; for fj in 0..2 { let bj=if fj==0 {2} else {0}; for j in 0..6 { expected += (a[(fi*6+i)*n+fj*6+j] as f32 as f64) * x[bj*6+j] as f64; } } }
            let got=y[b*6+i] as f64;
            if !got.is_finite() || (got-expected).abs() > 1e-5 * expected.abs().max(1.0) { return Err(Fault::Shape { kernel: "spmv.fixed_csr", detail: format!("GPU inverse mismatch at {}: {} vs {}", b*6+i,got,expected) }); }
        }}
        Ok(())
    })();
    let rx=xd.free(device); let ry=yd.free(device); let ri=inv.free(device);
    result?; rx?; ry?; ri?; Ok(())
}

fn spans_overlap(x: Handle, out: Handle, count: usize) -> bool {
    let bytes = count as u64 * 4;
    x.arena == out.arena && (x.off as u64) < out.off as u64 + bytes && (out.off as u64) < x.off as u64 + bytes
}



