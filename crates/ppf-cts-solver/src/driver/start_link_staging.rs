use std::borrow::Cow;
pub struct Lookup<'a> { pub mode: u32, pub offsets: Cow<'a, [u32]>, pub index: Cow<'a, [u32]> }
pub fn stage<'a>(offsets: &'a [u32], index: &'a [u32]) -> Option<Lookup<'a>> {
    if offsets.is_empty() || offsets[0] != 0 || *offsets.last()? as usize != index.len() ||
        offsets.windows(2).any(|w| w[0] > w[1]) { return None; }
    let mut ranges = 0usize;
    for w in offsets.windows(2) {
        let row = &index[w[0] as usize..w[1] as usize];
        if row.windows(2).any(|p| p[0] > p[1]) {
            return Some(Lookup { mode: 1, offsets: Cow::Borrowed(offsets), index: Cow::Borrowed(index) });
        }
        let mut previous = None;
        for &v in row {
            if previous.map_or(true, |p: u32| v > p.saturating_add(1)) {
                ranges = ranges.checked_add(1)?;
            }
            previous = Some(v);
        }
    }
    let words = ranges.checked_mul(2)?;
    if words >= index.len() || words > u32::MAX as usize {
        return Some(Lookup { mode: 2, offsets: Cow::Borrowed(offsets), index: Cow::Borrowed(index) });
    }
    let mut out = Vec::<u32>::new(); if out.try_reserve_exact(words).is_err() { return Some(Lookup { mode: 2, offsets: Cow::Borrowed(offsets), index: Cow::Borrowed(index) }); }
    let mut out_offsets = Vec::new(); if out_offsets.try_reserve_exact(offsets.len()).is_err() { return Some(Lookup { mode: 2, offsets: Cow::Borrowed(offsets), index: Cow::Borrowed(index) }); }
    out_offsets.push(0);
    for w in offsets.windows(2) {
        let start = out.len();
        for &value in &index[w[0] as usize..w[1] as usize] {
            if out.len() > start && value <= out[out.len()-1].saturating_add(1) {
                let last = out.len()-1;
                out[last] = value.max(out[last]);
            } else { out.push(value); out.push(value); }
        }
        out_offsets.push(u32::try_from(out.len()/2).ok()?);
    }
    Some(Lookup { mode: 3, offsets: Cow::Owned(out_offsets), index: Cow::Owned(out) })
}
#[cfg(test)]
mod tests {
 use super::*;
 #[test] fn compressed_and_fallback() {
  let x=stage(&[0,6], &[0,1,1,2,0x80000000,0x80000001]).unwrap();
  assert_eq!(x.mode,3); assert_eq!(&*x.index,&[0,2,0x80000000,0x80000001]);
  assert_eq!(stage(&[0,2], &[9,1]).unwrap().mode,1);
  assert_eq!(stage(&[0,2], &[1,9]).unwrap().mode,2);
  assert!(stage(&[0,3], &[1,2]).is_none());
 }
}


/// Verify that the selected backend actually implements all staged layouts.
pub fn verify_device<D: ppf_cts_compute::Device>(device: &mut D) -> Result<(), ppf_cts_compute::Fault> {
    use ppf_cts_compute::{AllocLabel, Buffer, Fault};
    use super::kernels::StartLinkCodecProbeArgs;
    let mut index=Buffer::<u32>::none(); let mut offset=Buffer::<u32>::none();
    let mut from=Buffer::<u32>::none(); let mut target=Buffer::<u32>::none();
    let mut output=Buffer::<u32>::none();
    let result = (|| -> Result<(), Fault> {
    index.size(device,12,AllocLabel("codec.probe.index"))?;
    offset.size(device,2,AllocLabel("codec.probe.offset"))?;
    from.size(device,2,AllocLabel("codec.probe.from"))?;
    target.size(device,2,AllocLabel("codec.probe.target"))?;
    output.size(device,2,AllocLabel("codec.probe.output"))?;
    from.write(device,0,&[0,0])?; target.write(device,0,&[8,11])?;
    for mode in 1..=3 {
        output.write(device,0,&[u32::MAX,u32::MAX])?;
        if mode==3 { index.write(device,0,&[4,10])?; offset.write(device,0,&[0,1])?; }
        else { index.write(device,0,&[0,2,4,6,8,10,12,14,16,18,20,22])?; offset.write(device,0,&[0,12])?; }
        let args=StartLinkCodecProbeArgs{index:index.handle(),offset:offset.handle(),mode,
            from_vertices:from.handle(),target_vertices:target.handle(),output:output.handle(),
            extent:2,count:2,seam_arena_count:0};
        unsafe { device.launch("start_link.codec.probe",&args,2)?; }
        let mut result=[0u32;2];
        output.read(device,0,&mut result)?;
        if result != [1,0] {
            return Err(Fault::Platform{call:"start-link codec capability",detail:format!("backend rejected mode {mode}")});
        }
    }
    Ok(())
    })();
    let mut cleanup = Ok(());
    for buffer in [&mut index, &mut offset, &mut from, &mut target, &mut output] {
        if !buffer.handle().is_none() {
            let freed = buffer.free(device).map_err(|error| Fault::Platform {
                call: "start-link probe cleanup",
                detail: error.to_string(),
            });
            if cleanup.is_ok() { cleanup = freed; }
        }
    }
    cleanup?;
    result

}


#[cfg(test)]
mod runtime_tests {
    #[test]
    fn actual_host_device_probe_accepts_all_modes() {
        let mut device = super::super::launch::host_device();
        super::verify_device(&mut device).expect("all codec probe modes should run");
    }
}
