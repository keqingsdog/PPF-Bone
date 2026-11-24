//! Optional compact readback of the current, already checked joint export.
use super::{bad, read, try_vec, Joint};
use crate::driver::kernels::{id, GatherPositionAbsoluteArgs};
use ppf_cts_compute::{AllocLabel, Buffer, Device, Fault, Handle};
use std::{mem::size_of, sync::OnceLock, time::Instant};

pub(super) const WIDTH: usize = 24;
pub(super) const PADDED_RECORD: usize = WIDTH * WIDTH;
const LANES: usize = 3;
const MAX_SCRATCH_BYTES: usize = 8 << 20;
const GATHER_REGION: &str = "pcg.joint_h.gather";
const INDEX_LABEL: AllocLabel = AllocLabel("body_capture.joint_h_indices");
const VALUE_LABEL: AllocLabel = AllocLabel("body_capture.joint_h_compact");

#[derive(Clone, Copy)]
enum Mode {
    Disabled,
    Gather,
    Slab,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("PPF_COMPACT_JOINT_H_EXPORT").as_deref() {
        Ok("1") => Mode::Gather,
        Ok("slab") => Mode::Slab,
        _ => Mode::Disabled,
    })
}

fn trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PPF_JOINT_H_EXPORT_TRACE").as_deref() == Ok("1"))
}

fn row_triples(width: usize) -> usize {
    width.div_ceil(LANES)
}

fn shape(counts: &[u32], padded: usize, limit: usize) -> Result<Option<usize>, Fault> {
    let mut triples = 0usize;
    for &count in counts {
        let count = count as usize;
        if count > WIDTH {
            return Err(bad("joint count"));
        }
        triples = triples.checked_add(count * row_triples(count)).ok_or_else(|| bad("extent"))?;
    }
    // Host and device copies of both streams, including device growth headroom.
    let scratch = triples.checked_mul((2 + 1) * (LANES + 1) * size_of::<u32>());
    // Prefer the old single read when packing saves less than a quarter of the
    // transferred bytes. This also keeps scratch below the original host mirror.
    let transfer = triples.checked_mul(LANES + 1);
    if padded / LANES > u32::MAX as usize
        || !scratch.is_some_and(|bytes| bytes <= limit)
        || !transfer.is_some_and(|words| words <= padded - padded / 4)
    {
        return Ok(None);
    }
    Ok(Some(triples))
}

fn indices(counts: &[u32], triples: usize) -> Option<Vec<u32>> {
    let mut indices = Vec::new();
    indices.try_reserve_exact(triples).ok()?;
    for (record, &count) in counts.iter().enumerate() {
        let count = count as usize;
        for row in 0..count {
            let start = (record * PADDED_RECORD + row * WIDTH) / LANES;
            indices.extend((start..start + row_triples(count)).map(|index| index as u32));
        }
    }
    Some(indices)
}

// Strip at most two unused lanes at the end of each copied row, in place.
// A multiple-of-three row is already packed and needs no copy.
fn strip_padding(counts: &[u32], values: &mut Vec<f32>) {
    let mut src = 0;
    let mut dst = 0;
    for &count in counts {
        let count = count as usize;
        let stride = LANES * row_triples(count);
        for _ in 0..count {
            if src != dst {
                values.copy_within(src..src + count, dst);
            }
            src += stride;
            dst += count;
        }
    }
    debug_assert_eq!(src, values.len());
    values.truncate(dst);
}

fn strip_padded_records(counts: &[u32], values: &mut Vec<f32>) -> Result<(), Fault> {
    if counts.iter().any(|&count| count as usize > WIDTH) {
        return Err(bad("joint count"));
    }
    let mut dst = 0;
    for (record, &count) in counts.iter().enumerate() {
        let count = count as usize;
        for row in 0..count {
            let src = record * PADDED_RECORD + row * WIDTH;
            if src != dst {
                values.copy_within(src..src + count, dst);
            }
            dst += count;
        }
    }
    values.truncate(dst);
    Ok(())
}

pub(super) fn views<'a>(counts: &[u32], ids: &'a [u32], h: &'a [f32]) -> Result<Vec<Joint<'a>>, Fault> {
    let mut views = try_vec(counts.len())?;
    let mut first = 0;
    for (&count, record_ids) in counts.iter().zip(ids.chunks_exact(WIDTH)) {
        let count = count as usize;
        let last = first + count * count;
        views.push(Joint {
            ids: record_ids.get(..count).ok_or_else(|| bad("joint count"))?,
            h: h.get(first..last).ok_or_else(|| bad("joint h extent"))?,
        });
        first = last;
    }
    if views.len() != counts.len() || first != h.len() {
        return Err(bad("joint extent"));
    }
    Ok(views)
}

fn gather(
    d: &mut impl Device,
    h: Handle,
    padded: usize,
    input: &[u32],
    values: &mut [f32],
) -> Result<bool, Fault> {
    let source_triples = padded / LANES;
    if input.iter().any(|&index| index as usize >= source_triples) {
        return Err(bad("joint gather index"));
    }
    let mut indices = Buffer::<u32>::none();
    let mut output = Buffer::<f32>::none();
    let result: Result<bool, Fault> = (|| {
        // Allocation refusal is optional. Failed fill, transfer, launch or free
        // is still a device fault and must never be retried as a padded read.
        match indices.size(d, input.len(), INDEX_LABEL) {
            Err(Fault::Alloc { .. }) => return Ok(false),
            other => other?,
        }
        match output.size(d, values.len(), VALUE_LABEL) {
            Err(Fault::Alloc { .. }) => return Ok(false),
            other => other?,
        }
        indices.write(d, 0, input)?;
        let args = GatherPositionAbsoluteArgs {
            curr: h,
            indices: indices.handle(),
            vertex_count: source_triples as u32,
            out: output.handle(),
            count: input.len() as u32,
            seam_arena_count: 0,
        };
        // The generated entry copies exactly three opaque lanes. Layout bounds
        // originate from the validated export counts, and it checks the triple
        // index again on the device. Any copied trailing lanes are discarded.
        unsafe { d.launch(GATHER_REGION, &args, args.count)? };
        output.read(d, 0, values)?;
        Ok(true)
    })();
    let output_cleanup = output.free(d);
    let indices_cleanup = indices.free(d);
    let copied = result?;
    output_cleanup?;
    indices_cleanup?;
    Ok(copied)
}

pub(super) fn read_hessian(
    d: &mut impl Device,
    h: Handle,
    counts: &[u32],
) -> Result<Option<Vec<f32>>, Fault> {
    let start = trace_enabled().then(Instant::now);
    match mode() {
        Mode::Disabled => return Ok(None),
        Mode::Slab => {
            let mut values = read_padded(d, h, counts.len())?;
            strip_padded_records(counts, &mut values)?;
            if let Some(start) = start {
                eprintln!(
                    "PPF_JOINT_H_SLAB records={} active_values={} elapsed_us={}",
                    counts.len(), values.len(), start.elapsed().as_micros(),
                );
            }
            return Ok(Some(values));
        }
        Mode::Gather => (),
    }
    let padded = counts.len().checked_mul(PADDED_RECORD).ok_or_else(|| bad("extent"))?;
    if padded > h.size as usize {
        return Err(bad("read extent"));
    }
    let Some(triples) = shape(counts, padded, MAX_SCRATCH_BYTES)? else {
        return Ok(None);
    };
    if triples == 0 {
        return Ok(Some(Vec::new()));
    }
    if d.missing().contains(&id::GATHER_POSITION_ABSOLUTE) {
        return Ok(None);
    }
    let Some(input) = indices(counts, triples) else {
        return Ok(None);
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(triples * LANES).is_err() {
        return Ok(None);
    }
    values.resize(triples * LANES, 0.0);
    if !gather(d, h, padded, &input, &mut values)? {
        return Ok(None);
    }
    let read_bytes = values.len() * size_of::<f32>();
    strip_padding(counts, &mut values);
    if let Some(start) = start {
        eprintln!(
            "PPF_JOINT_H_EXPORT route=compact records={} original_bytes={} read_bytes={read_bytes} index_bytes={} active_values={} elapsed_us={}",
            counts.len(), padded * size_of::<f32>(), input.len() * size_of::<u32>(), values.len(),
            start.elapsed().as_micros(),
        );
    }
    Ok(Some(values))
}

pub(super) fn read_padded(d: &mut impl Device, h: Handle, records: usize) -> Result<Vec<f32>, Fault> {
    let start = trace_enabled().then(Instant::now);
    let values = read::<f32>(d, h, records.checked_mul(PADDED_RECORD).ok_or_else(|| bad("extent"))?)?;
    if let Some(start) = start {
        eprintln!(
            "PPF_JOINT_H_EXPORT route=padded records={records} original_bytes={} read_bytes={} index_bytes=0 elapsed_us={}",
            values.len() * size_of::<f32>(), values.len() * size_of::<f32>(), start.elapsed().as_micros(),
        );
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;

    fn raw_h(records: usize) -> Vec<f32> {
        let patterns = [0x0000_0000, 0x8000_0000, 0x0000_0007, 0x7f80_0000, 0x7fc9_1234];
        (0..records * PADDED_RECORD)
            .map(|index| f32::from_bits(if index % 7 < patterns.len() {
                patterns[index % 7]
            } else {
                0x3f80_0000 | index as u32
            }))
            .collect()
    }

    fn expected_h(padded: &[f32], counts: &[u32]) -> Vec<u32> {
        let mut output = Vec::new();
        for (record, &count) in counts.iter().enumerate() {
            let width = count as usize;
            for row in 0..width {
                let first = record * PADDED_RECORD + row * WIDTH;
                output.extend(padded[first..first + width].iter().map(|lane| lane.to_bits()));
            }
        }
        output
    }

    #[test]
    fn all_valid_widths_and_duplicate_ids_borrow_the_original_bits() {
        let counts: Vec<_> = (0..=WIDTH as u32).chain([24, 12, 0, 22, 3]).collect();
        let padded = raw_h(counts.len());
        let expected = expected_h(&padded, &counts);
        let count = shape(&counts, padded.len(), MAX_SCRATCH_BYTES).unwrap().unwrap();
        let input = indices(&counts, count).unwrap();
        assert_eq!(input.len(), count);

        let mut device = host_device();
        let mut source = Buffer::<f32>::none();
        source.size(&mut device, padded.len(), AllocLabel("test.joint_h")).unwrap();
        source.write(&mut device, 0, &padded).unwrap();
        let mut actual = vec![0.0; count * LANES];
        assert!(gather(&mut device, source.handle(), padded.len(), &input, &mut actual).unwrap());
        strip_padding(&counts, &mut actual);
        assert_eq!(actual.iter().map(|lane| lane.to_bits()).collect::<Vec<_>>(), expected);

        let ids: Vec<_> = (0..counts.len() * WIDTH).map(|i| (i % 17) as u32).collect();
        let items = views(&counts, &ids, &actual).unwrap();
        for (record, (&count, joint)) in counts.iter().zip(&items).enumerate() {
            assert_eq!(joint.ids, &ids[record * WIDTH..record * WIDTH + count as usize]);
            assert_eq!(joint.h.len(), (count * count) as usize);
        }
        source.free(&mut device).unwrap();
    }

    #[test]
    fn padded_slab_has_the_same_active_values_and_faults_on_invalid_count() {
        let counts = [0, 24, 1, 19, 12, 3, 0];
        let mut padded = raw_h(counts.len());
        let expected = expected_h(&padded, &counts);
        strip_padded_records(&counts, &mut padded).unwrap();
        assert_eq!(padded.iter().map(|lane| lane.to_bits()).collect::<Vec<_>>(), expected);
        assert!(strip_padded_records(&[25], &mut padded).is_err());
        assert!(views(&[1], &[0; WIDTH], &[]).is_err());
    }

    #[test]
    fn layout_admission_refuses_dense_overbudget_and_bad_counts() {
        assert_eq!(shape(&[], 0, 0).unwrap(), Some(0));
        assert_eq!(shape(&[0], PADDED_RECORD, 0).unwrap(), Some(0));
        assert_eq!(shape(&[24], PADDED_RECORD, MAX_SCRATCH_BYTES).unwrap(), None);
        assert_eq!(shape(&[12], PADDED_RECORD, 1).unwrap(), None);
        assert!(shape(&[25], PADDED_RECORD, MAX_SCRATCH_BYTES).is_err());
    }

    #[test]
    fn invalid_gather_index_is_rejected_before_native_launch() {
        let mut device = host_device();
        let mut source = Buffer::<f32>::none();
        source.size(&mut device, PADDED_RECORD, AllocLabel("test.joint_h")).unwrap();
        let mut output = [0.0; LANES];
        let result = gather(&mut device, source.handle(), PADDED_RECORD, &[(PADDED_RECORD / LANES) as u32], &mut output);
        assert!(result.unwrap_err().to_string().contains("joint gather index"));
        source.free(&mut device).unwrap();
    }
}
