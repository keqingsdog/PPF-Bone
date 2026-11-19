//! Resident fixed CSR reduced operator. Caller must explicitly free on the owning device.
use ppf_cts_compute::{AllocLabel, Buffer, Device, Fault, Handle, Pod};
use super::reduced_operator_pack::Packed;
use super::spmv::{fixed_csr_spmv, FixedCsrView};

pub(super) struct ReducedOperatorDevice {
    index: Buffer<u32>, offset: Buffer<u32>, value: Buffer<f32>,
    transpose_pair: Buffer<u32>, transpose_offset: Buffer<u32>, rows: u32, live: bool,
}
fn upload<T: Pod>(device: &mut impl Device, dst: &mut Buffer<T>, data: &[T], label: &'static str) -> Result<(), Fault> {
    dst.size(device, data.len().max(1), AllocLabel(label))?;
    if !data.is_empty() { dst.write(device, 0, data)?; }
    Ok(())
}
impl ReducedOperatorDevice {
    pub fn upload(device: &mut impl Device, p: &Packed) -> Result<Self, Fault> {
        let mut d = Self { index: Buffer::none(), offset: Buffer::none(), value: Buffer::none(), transpose_pair: Buffer::none(), transpose_offset: Buffer::none(), rows: p.rows, live: true };
        let result = (|| { upload(device, &mut d.index, &p.index, "reduced_operator.index")?; upload(device, &mut d.offset, &p.offset, "reduced_operator.offset")?; upload(device, &mut d.value, &p.value, "reduced_operator.value")?; upload(device, &mut d.transpose_pair, &[0u32], "reduced_operator.transpose_pair")?; upload(device, &mut d.transpose_offset, &p.transpose_offset, "reduced_operator.transpose_offset")?; Ok(()) })();
        if let Err(e) = result { let _ = d.free(device); return Err(e); }
        Ok(d)
    }
    /// Call with distinct complete reduced f32 spans on the same device. Both allocations and
    /// this object must remain live until the device completes the dispatch.
    pub unsafe fn apply(&self, device: &mut impl Device, x: Handle, out: Handle) -> Result<(), Fault> {
        if !self.live { return Err(Fault::Shape { kernel: "spmv.fixed_csr", detail: "reduced operator has been freed".into() }); }
        let count = 3 * self.rows as usize;
        if x.size as usize != count || out.size as usize != count || x.off % 4 != 0 || out.off % 4 != 0 || spans_overlap(x,out,count) { return Err(Fault::Shape { kernel: "spmv.fixed_csr", detail: "invalid reduced spans or overlap".into() }); }
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


fn spans_overlap(x: Handle, out: Handle, count: usize) -> bool {
 let bytes = count as u64 * 4;
 x.arena == out.arena && (x.off as u64) < out.off as u64 + bytes && (out.off as u64) < x.off as u64 + bytes
}


