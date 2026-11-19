// Candidate fragment for transactional device upload. Kept outside the tested tree.
use ppf_cts_compute::{AllocLabel, Buffer, Device, Fault};

fn reclaim<D: Device>(device: &mut D, first: &mut Buffer<u32>, second: &mut Buffer<u32>) -> Result<(), Fault> {
    let mut result = Ok(());
    for buffer in [first, second] {
        if !buffer.handle().is_none() {
            let freed = buffer.free(device).map_err(|error| Fault::Platform {
                call: "start-link scratch cleanup",
                detail: error.to_string(),
            });
            if result.is_ok() { result = freed; }
        }
    }
    result
}

fn upload<D: Device>(device: &mut D, buffer: &mut Buffer<u32>, host: &[u32], label: &'static str) -> Result<(), Fault> {
    buffer.size(device, host.len(), AllocLabel(label))?;
    if !host.is_empty() { buffer.write(device, 0, host)?; }
    Ok(())
}

/// The caller validates offsets, vertex domain, and capabilities before entry.
/// New allocations are private until both uploads finish. Published raw input is
/// never mutated. Scratch cleanup errors are fatal, including on Alloc.
pub fn publish<D: Device>(
    device: &mut D, index: &mut Buffer<u32>, offset: &mut Buffer<u32>,
    raw_index: &[u32], raw_offset: &[u32], packed: &super::start_link_staging::Lookup<'_>,
) -> Result<u32, Fault> {
    if packed.mode == 3 {
        let mut new_index = Buffer::<u32>::none();
        let mut new_offset = Buffer::<u32>::none();
        let prepared = (|| {
            upload(device, &mut new_index, &packed.index, "start_link.codec.index")?;
            upload(device, &mut new_offset, &packed.offsets, "start_link.codec.offset")?;
            Ok::<(), Fault>(())
        })();
        match prepared {
            Ok(()) => {
                // Caller is in initialization, with no live users of these buffers.
                // Publish the pair only after both uploads complete.
                std::mem::swap(index, &mut new_index);
                std::mem::swap(offset, &mut new_offset);
                reclaim(device, &mut new_index, &mut new_offset)?;
                return Ok(3);
            }
            Err(fault) => {
                reclaim(device, &mut new_index, &mut new_offset)?;
                if !matches!(fault, Fault::Alloc { .. }) { return Err(fault); }
            }
        }
    }
    // Preserve sorted-vs-unsorted mode even when compressed allocation fails.
    let mode = if packed.mode == 1 { 1 } else { 2 };
    upload(device, index, raw_index, "start_link.index")?;
    upload(device, offset, raw_offset, "start_link.offset")?;
    Ok(mode)
}
