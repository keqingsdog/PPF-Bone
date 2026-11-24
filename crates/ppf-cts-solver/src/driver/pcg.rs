// Fork modification: changed from the upstream baseline.
// File: crates/ppf-cts-solver/src/driver/pcg.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Preconditioned conjugate gradient: the iteration control.
//!
//! Every value comes from a shared body: the operator apply through
//! `spmv.kernel.cpp`, the preconditioner through `block_jacobi.kernel.cpp`, the
//! vector updates through `vec_ops.kernel.cpp`. What lives here is the loop, the
//! convergence test, the guards and the sequencing, none of which computes a
//! value that any other backend also computes.
//!
//! # Two rules inherited whole, because each has shipped as a defect once
//!
//! **The relative-residual denominator is the SEEDED initial residual
//! `||b - A x0||_1`, never `||b||_1`.** The solver never enters with a zero `x`:
//! the driver pre-loads every prescribed (Dirichlet) row of `dx` with its exact
//! correction before the solve. Those rows carry the pin-barrier stiffness, so
//! with any kinematically driven collider they dominate `||b||_1` by orders of
//! magnitude while saying nothing about the dynamic degrees of freedom. Switching
//! to `||b||_1` makes the tolerance exceed the whole dynamic residual, so PCG
//! "converges" in a few iterations and the stiff membrane stretch modes never
//! resolve. It shipped once and was reverted; it must not come back.
//!
//! **The curvature guard fires on a SIGN, and that sign is only real above the
//! round-off of its own sum.** `p^T A p` sums signed fp32 contributions whose
//! magnitudes can exceed the curvature by more than fp32 epsilon, so a small
//! negative value is noise rather than an indefinite operator. A value negative
//! BEYOND the bound aborts; within it, the curvature is unresolvable and the
//! iteration truncates and returns its iterate. It is a round-off bound, not a
//! loosened guard: a genuine assembly defect puts the Rayleigh quotient at order
//! one, six orders above it.
//!
//! # Where the bound comes from, and which scale it is measured over
//!
//! The verdict itself is `pcg_alpha` (`src/kernels/solver/pcg.kernel.cpp`), one
//! shared body every path reaches, so the classification into "proceed",
//! "truncate" and "not positive definite" is not spelled twice. What IS the
//! caller's business is the SCALE handed to `pcg_curvature_noise`, and there
//! are exactly two, chosen by whether the operator apply produced a
//! `sum|contribution|` beside its dot:
//!
//! - a FUSED apply, which accumulates `sum|contribution|` over the very same
//!   signed terms that form `p^T A p`, passes that sum. The device-resident
//!   loop below takes this one, through [`slot::ABSDOT`].
//! - an apply that produces no such sum passes the Cauchy-Schwarz surrogate
//!   `|p|_2 |Ap|_2 >= sum_i |p_i . (Ap)_i|`, formed from three plain inner
//!   products. The locked and PDRD-reduced solves take this one, through
//!   [`DotTerms::norm_product`].
//!
//! The surrogate is the LOOSER of the two, and that is why it is admissible
//! where the fused sum is unavailable: it errs toward truncating rather than
//! aborting.
//!
//! **THE PER-ROW MAGNITUDE BESIDE THE DOT IS NOT A THIRD OPTION.**
//! `pcg_dot_terms` sums a row's three components BEFORE taking the absolute
//! value, so `sum_i |p_i . (Ap)_i|` has already spent every cancellation
//! within a row and is never larger than either quantity above. Reducing it
//! and calling it `absolute_dot` makes the bound TIGHTER than the two the
//! shared verdict is written against, which moves marginal iterates from
//! truncating quietly to aborting loudly. That is the wrong direction and it is
//! not a conservative choice: the whole `pAp<=0` triage keys on the distinction
//! between an unresolvable curvature, which truncates Steihaug-style and is not
//! an error, and a genuine assembly defect, which clears the bound by six
//! orders and must abort. Narrowing the bound erases exactly the first
//! category. The magnitude array stays because `pcg_dot_terms` is a shared body
//! producing both numbers; nothing here judges a curvature by it.

// The solves are called from `step.rs`. The allow is for what no dispatch path
// reaches, including `build_block_jacobi` and `block_jacobi_failure`, which
// only this module's own tests call, and `pcg_beta`, `encode_combine_in_place`
// and `DotTerms::dot_with_bound`, which nothing calls.
#![allow(dead_code)]

#[path = "no_progress.rs"]
mod no_progress;

#[path = "runtime_auto_policy.rs"]
pub(super) mod runtime_auto_policy;
#[path = "full_body_factor.rs"]
mod full_body_factor;
#[path = "full_body_native.rs"]
mod full_body_native;
use ppf_cts_compute::{AllocLabel, Buffer, Device, Encoder, EncoderExt, Fault, ReadbackBuffer};
use super::kernels::{
    Mat3MulArgs, PcgAlphaTermsArgs, PcgBetaResidentArgs,
    PcgBetaTermsArgs, PcgDotTermsArgs, VecAddScaledArgs, VecAddScaledIndirectArgs,
    PcgFoldAlphaArgs, PcgFoldBetaArgs, PcgRigidGroupL1Args,
    PcgUpdateRowFoldedArgs,
    VecBlockSumAbsCooperativeArgs, VecBlockSumCooperativeArgs,
    VecBlockSumDualCooperativeArgs, VecBlockSumPairCooperativeArgs,
    VecCombineArgs,
    VecCombineIndirectArgs,
};
// Named only by the test that drives the resident alpha step on its own.
#[cfg(test)]
use super::kernels::PcgAlphaResidentArgs;
use super::operator::Operator;
use super::reduce;

/// The `cause` codes `pcg_alpha` and `pcg_beta` classify their scalar
/// with, mirroring the anonymous enum in `src/kernels/solver/pcg.kernel.cpp`. Named
/// here rather than compared as bare integers so a reordering there is a
/// mismatch a reader can see.
const PCG_BREAK_NONE: i32 = 0;
const PCG_BREAK_PAP: i32 = 1;
const PCG_BREAK_RZ: i32 = 2;
const PCG_BREAK_NOISE: i32 = 3;

extern "C" {
    // THE ONE SHIM THIS MODULE STILL NAMES DIRECTLY, and it is named here so the
    // exception is visible rather than buried.
    //
    // `block_jacobi_invert` is a PER-BLOCK helper with a return value, called
    // in a host loop that stops at the FIRST refused row so the report can name
    // it. The seam dispatches over a range and returns no value, so putting this
    // behind it needs a `_range` shim plus a first-failure reduction over the
    // rows, which is a change to the C++ side rather than to the driver. Until
    // that lands this is a direct call, and it is the only one left in the linear
    // solve.
    fn block_jacobi_invert_abi(
        block: *const f32,
        inverse: *mut f32,
        lambda_max_out: *mut f32,
    ) -> i32;
}

/// `destination += scale * source`, elementwise, through the shared body.
///
/// The body is `vec_add_scaled` in `primitives/vec_ops.kernel.cpp`, which CUDA
/// and Metal compile from the same bytes. Writing `destination[i] += scale *
/// source[i]` here in Rust instead would be a second implementation of it, and
/// not a harmless one: the host renderings are compiled with
/// `-ffp-contract=fast` so their arithmetic matches a contracting device, and
/// Rust never contracts, so the two spellings can differ in the last bit.
///
/// # Safety
/// Both slices must outlive the dispatch this appends.
unsafe fn encode_add_scaled(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    scale: f32,
    n: usize,
) -> Result<(), Fault> {
    if n == 0 {
        return Ok(());
    }
    let args = VecAddScaledArgs {
        source,
        destination,
        scale,
        // The in-kernel guard bound, which a generated entry point carries in
        // its record because Metal never faults on an out-of-bounds access and
        // every backend rounds a launch up to whole threadgroups. It is the
        // same `n` the extent below takes, and `Encoder::elements` faults if
        // the two ever differ.
        count: n as u32,
        seam_arena_count: 0,
    };
    encoder.elements(&args, n as u32)
}

/// `destination = a * source_a + b * source_b`, elementwise, through the shared
/// body `vec_combine`.
///
/// # Safety
/// The three references must each name `n` floats that outlive the dispatch.
/// `source_b` MAY alias `destination`, which is the `p = z + beta p` case: the
/// body reads element `i` before writing element `i` and every chunk is a
/// disjoint span of indices. `source_a` must not alias `destination`.
unsafe fn encode_combine_raw(
    encoder: &mut dyn Encoder,
    source_a: ppf_cts_compute::Handle,
    source_b: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    a: f32,
    b: f32,
    n: usize,
) -> Result<(), Fault> {
    if n == 0 {
        return Ok(());
    }
    let args = VecCombineArgs {
        source_a,
        source_b,
        destination,
        a,
        b,
        count: n as u32,
        seam_arena_count: 0,
    };
    encoder.elements(&args, n as u32)
}

/// `destination = a * source_a + b * source_b` for three distinct buffers.
fn combine<D: Device>(
    device: &mut D,
    source_a: ppf_cts_compute::Handle,
    source_b: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    a: f32,
    b: f32,
    n: usize,
) -> Result<(), Fault> {
    let refs = (source_a, source_b, destination);
    device.run("pcg.combine", |encoder| {
        // Safety: three distinct slices, each `n` long, so nothing aliases, and
        // all three are borrowed for the whole call.
        unsafe { encode_combine_raw(encoder, refs.0, refs.1, refs.2, a, b, n) }
    })?;
    Ok(())
}

// Warm candidates use separate scratch so source_a never aliases destination.
fn warm_difference<D: Device>(device: &mut D, current: ppf_cts_compute::Handle,
    cold: ppf_cts_compute::Handle, out: ppf_cts_compute::Handle, n: usize) -> Result<(), Fault> {
    combine(device, current, cold, out, 1.0, -1.0, n)
}

fn warm_add_second<D: Device>(device: &mut D, candidate: ppf_cts_compute::Handle,
    history: ppf_cts_compute::Handle, out: ppf_cts_compute::Handle,
    coefficient: f32, n: usize) -> Result<(), Fault> {
    combine(device, candidate, history, out, 1.0, coefficient, n)
}

fn warm_mask_history<D: Device>(device: &mut D, history: ppf_cts_compute::Handle,
    free_row: ppf_cts_compute::Handle, n: usize) -> Result<(), Fault> {
    let args = super::kernels::PcgWarmMaskHistoryArgs {
        history,
        free_row,
        count: n as u32,
        seam_arena_count: 0,
    };
    unsafe { device.launch("pcg.warm_mask_history", &args, n as u32).map(|_| ()) }
}

/// Values per block in the DEVICE fold, and the same contract
/// [`super::reduce`] states for the host folds that remain.
///
/// It is stated in ONE place and handed to the kernel as an argument, because
/// two statements of it are two fold shapes that can disagree, and the shape
/// decides the last bits of every scalar this recurrence carries.
const FOLD_WIDTH: usize = super::reduce::BLOCK;

/// Threads per group in the COOPERATIVE fold, and the reason it is not
/// `FOLD_WIDTH`.
///
/// `FOLD_WIDTH` is how many VALUES a block folds; this is how many THREADS share
/// that block. The two were one number while a block was one thread's serial
/// walk, which is exactly what made the first level of a fold over
/// `3 * vertices` launch 915 threads and cost 53 percent of GPU time.
/// Separating them is the fix: the block keeps its width, so the LEVEL COUNT
/// and the values each level folds are unchanged, and 64 lanes share it, so the
/// level's thread count is 64 times what it was.
const FOLD_LANES: u32 = 64;

/// Where each scalar of the recurrence lives in the workspace's scalar buffer.
///
/// ONE BUFFER RATHER THAN TEN. Every scalar of the recurrence is carved out of
/// a single allocation, so a scheduled check reads a window of them with one
/// copy. A separate allocation per scalar would be ten handles to bind and ten
/// transfers to read.
mod slot {
    /// `r . z` from the PREVIOUS iteration, which alpha divides and the beta
    /// step rolls forward.
    pub const RZ0: usize = 0;
    /// `r . z` at the current iterate.
    pub const RZ1: usize = 1;
    pub const P_AP: usize = 2;
    pub const ALPHA: usize = 3;
    pub const BETA: usize = 4;
    /// `||r||_1`, the only scalar the host reads every iteration.
    pub const ERR: usize = 5;
    /// The scale the curvature bound is taken from: the sum, over the whole
    /// vector, of the MAGNITUDES of the signed contributions that add up to
    /// `p^T A p`. The operator forms it per row while it multiplies and the
    /// caller folds it here, so the bound and the curvature it bounds come out
    /// of one pass over the same terms.
    pub const ABSDOT: usize = 6;
    /// The offending scalar LATCHED at the iterate that broke, written by the
    /// resident alpha and beta bodies and never overwritten afterwards. It is a
    /// curvature for `PCG_BREAK_PAP` and an `r . z` for `PCG_BREAK_RZ`.
    ///
    /// It exists because the host samples on a STRIDE: by the time a scheduled
    /// check runs, `P_AP` holds a later iterate's value, and reporting that
    /// describes a different iterate entirely.
    pub const BREAK_VALUE: usize = 7;
    /// The round-off floors the two verdict bodies report beside their value.
    /// Nothing reads them today; they are bound because the bodies yield them
    /// and a gathered output has to name an allocation.
    pub const ALPHA_NOISE: usize = 8;
    pub const BETA_NOISE: usize = 9;
    /// NON-ZERO ONCE EITHER GUARD HAS LATCHED, and nothing more than that.
    ///
    /// The cause and the iteration it fired at are INTS and live in their own
    /// allocation, so reading them cost a second device-to-host copy on every
    /// batch, healthy or not, and they were measured at 939 of 2,404 such
    /// copies on `drape`. This float rides the scalar buffer the probe already
    /// downloads, so the int channel is read only when it holds something.
    ///
    /// IT IS NOT THE CAUSE, AND MUST NOT BECOME IT. Storing the enum here would
    /// buy the same copy and put a typed channel into a float, which is what
    /// the separate allocation exists to avoid; this answers only whether there
    /// is anything to read.
    pub const BREAK_FIRED: usize = 10;
    /// Where a ONE-OFF device fold lands when the host reads its result
    /// immediately. `DotTerms::dot` folds the per-row products here rather
    /// than downloading the whole vector and summing it on the host, which
    /// would relocate the reduction rather than transport its result. It is
    /// reused by every such fold because each is read before the next is
    /// encoded.
    pub const FOLD_SCRATCH: usize = 11;
    pub const FACTOR_FAULT: usize = 12;
    pub const COUNT: usize = 13;
}

/// Where each verdict's cause lives, which is a SECOND buffer because a cause
/// is an `int`.
///
/// Packing the cause into the float block and reinterpreting the bytes would
/// save a read; this driver keeps the types honest and pays one more small read
/// per iteration for it.
mod cause_slot {
    pub const ALPHA: usize = 0;
    pub const BETA: usize = 1;
    /// The STICKY breakdown cause, written by whichever of the two bodies
    /// breaks FIRST and never cleared inside a solve. Without it a strided
    /// read would miss a breakdown that
    /// a later healthy iteration overwrote, and the solve would continue on a
    /// direction the guard had already rejected.
    pub const BREAK_CAUSE: usize = 2;
    /// The iteration index the latch fired at, so the report names the
    /// offending iterate rather than the check that noticed it.
    pub const BREAK_ITER: usize = 3;
    /// The RUNNING ITERATION COUNT, advanced on the device by the alpha pass.
    ///
    /// It lives on the device rather than in a host variable, and the reason is
    /// not bookkeeping: an
    /// iteration whose arguments differ between two turns of the loop cannot be
    /// RECORDED ONCE AND REPLAYED, and a by-value index was the only argument
    /// that differed. It lives in this int block rather than the float one
    /// beside `BREAK_ITER`, which reads it.
    pub const ITERATION: usize = 4;
    pub const COUNT: usize = 5;
}

/// Floats the fold needs for its intermediate levels, folding `count` values.
///
/// The LAST level writes into the caller's scalar slot rather than into the
/// scratch, so a fold short enough to finish in one level needs none of this.
fn fold_scratch_len(count: usize) -> usize {
    let mut total = 0usize;
    let mut remaining = count;
    loop {
        let blocks = remaining.div_ceil(FOLD_WIDTH);
        if blocks <= 1 {
            return total;
        }
        total += blocks;
        remaining = blocks;
    }
}

/// Fold `source[..count]` to the single float `out` names, entirely on the
/// device.
///
/// THE SHAPE, WHICH IS A CONTRACT. Each level cuts its input into blocks of
/// [`FOLD_WIDTH`], sums each block serially in ascending index order, and
/// writes the block totals; the next level folds those the same way, until one
/// element remains. Kernel completion is the barrier between levels, which is
/// what lets a reduction exist at all on a seam with no cooperative launch
/// shape: `crates/ppf-cts-compute/src/device.rs` records that a reduction
/// reaches the CPU backend only as a body written without a threadgroup
/// barrier.
///
/// IT AGREES WITH [`super::reduce::sum`] BIT FOR BIT UP TO
/// `FOLD_WIDTH * FOLD_WIDTH` VALUES: blocks summed in index order, then the
/// block totals summed serially in index order. Past that length this takes
/// another level where a single serial pass would keep accumulating, which is a
/// MORE accurate association rather than a different rule, and it is the
/// association a strided device fold takes.
///
/// `absolute` applies to the FIRST level only, and that is not a convenience:
/// every level above it folds block totals that are already non-negative, so
/// taking magnitudes again would be a second absolute value over the same
/// numbers.
///
/// # Safety
/// `source`, `out` and the scratch must outlive the dispatches this appends,
/// and `out` must name exactly one float.
unsafe fn encode_fold(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    count: usize,
    scratch: &Buffer<f32>,
    out: ppf_cts_compute::Handle,
    absolute: bool,
) -> Result<(), Fault> {
    debug_assert!(count > 0, "a fold of nothing has no level to dispatch");
    let mut input = source;
    let mut remaining = count;
    let mut absolute = absolute;
    let mut cursor = 0usize;
    loop {
        let blocks = remaining.div_ceil(FOLD_WIDTH);
        let destination = if blocks == 1 {
            out
        } else {
            scratch.span(cursor, blocks)
        };
        if absolute {
            let args = VecBlockSumAbsCooperativeArgs {
                source: input,
                length: remaining as u32,
                width: FOLD_WIDTH as u32,
                total: destination,
                count: blocks as u32,
                seam_arena_count: 0,
            };
            encoder.groups(
                &args,
                blocks as u32,
                FOLD_LANES,
            )?;
        } else {
            let args = VecBlockSumCooperativeArgs {
                source: input,
                length: remaining as u32,
                width: FOLD_WIDTH as u32,
                total: destination,
                count: blocks as u32,
                seam_arena_count: 0,
            };
            encoder.groups(
                &args,
                blocks as u32,
                FOLD_LANES,
            )?;
        }
        if blocks == 1 {
            return Ok(());
        }
        cursor += blocks;
        input = destination;
        remaining = blocks;
        absolute = false;
    }
}

/// Fold TWO arrays of the same length to two scalars, in one dispatch chain.
///
/// EVERY LEVEL IS ONE DISPATCH INSTEAD OF TWO, and nothing else changes: each
/// output is the block sum of its own array over the same [`FOLD_WIDTH`] and in
/// the same ascending order [`encode_fold`] uses, so both scalars are
/// bit-identical to what two separate chains would produce and the curvature
/// bound stated against that shape still holds. It changes the launch count and
/// nothing else.
///
/// THE TWO ARRAYS NEED SEPARATE SCRATCH, because a level reads both and writes
/// both, so one region cannot serve. `second_base` is where the second array's
/// levels begin, and [`State::size_for`] sizes the buffer for both.
///
/// # Safety
/// Every buffer must outlive the dispatches this appends, and both outputs must
/// name exactly one float.
/// Fold TWO arrays of one length to two scalars in ONE dispatch.
///
/// [`encode_fold_pair`] walks a fixed-width tree because it starts from a
/// FULL-LENGTH array and no single group should own `rows` elements. The
/// partials the folded apply writes are one per group, so the tree has nothing
/// left to do: a single group whose window is the whole array gives each lane a
/// contiguous run of it, and the two outputs are folded by that one group.
///
/// # Safety
/// Every buffer must outlive the dispatch this appends, and both outputs must
/// name exactly one float.
unsafe fn encode_fold_pair_once(
    encoder: &mut dyn Encoder,
    first_source: ppf_cts_compute::Handle,
    second_source: ppf_cts_compute::Handle,
    count: usize,
    first_out: ppf_cts_compute::Handle,
    second_out: ppf_cts_compute::Handle,
) -> Result<(), Fault> {
    debug_assert!(count > 0, "a fold of nothing has no group to dispatch");
    let args = VecBlockSumPairCooperativeArgs {
        first_source,
        second_source,
        length: count as u32,
        // THE WINDOW IS THE WHOLE ARRAY, which is what makes this one group:
        // `vec_lane_run` splits `width` across the lanes, so a width equal to
        // the length leaves nothing for a second group to own.
        width: count as u32,
        first_total: first_out,
        second_total: second_out,
        count: 1,
        seam_arena_count: 0,
    };
    encoder.groups(&args, 1, FOLD_LANES)
}

unsafe fn encode_fold_pair(
    encoder: &mut dyn Encoder,
    first_source: ppf_cts_compute::Handle,
    second_source: ppf_cts_compute::Handle,
    count: usize,
    scratch: &Buffer<f32>,
    second_base: usize,
    first_out: ppf_cts_compute::Handle,
    second_out: ppf_cts_compute::Handle,
) -> Result<(), Fault> {
    debug_assert!(count > 0, "a fold of nothing has no level to dispatch");
    let mut first_in = first_source;
    let mut second_in = second_source;
    let mut remaining = count;
    let mut cursor = 0usize;
    loop {
        let blocks = remaining.div_ceil(FOLD_WIDTH);
        let (first_dst, second_dst) = if blocks == 1 {
            (first_out, second_out)
        } else {
            (
                scratch.span(cursor, blocks),
                scratch.span(second_base + cursor, blocks),
            )
        };
        let args = VecBlockSumPairCooperativeArgs {
            first_source: first_in,
            second_source: second_in,
            length: remaining as u32,
            width: FOLD_WIDTH as u32,
            first_total: first_dst,
            second_total: second_dst,
            count: blocks as u32,
            seam_arena_count: 0,
        };
        encoder.groups(
            &args,
            blocks as u32,
            FOLD_LANES,
        )?;
        if blocks == 1 {
            return Ok(());
        }
        cursor += blocks;
        first_in = first_dst;
        second_in = second_dst;
        remaining = blocks;
    }
}

/// Fold TWO arrays of DIFFERENT lengths to two scalars, in one dispatch chain.
///
/// [`encode_fold_pair`] shares one length and one block count between its two
/// arrays, which is the honest restriction for it: the curvature and its bound
/// are two views of one pass and are folded in lockstep. THE TWO CHAINS HERE
/// ARE NOT IN LOCKSTEP. `||r||_1` runs over `3 * rows` floats and `r . z` over
/// `rows` of them, so their levels have different extents and one reaches a
/// single element before the other does.
///
/// EACH CHAIN KEEPS ITS OWN CURSOR, EXTENT AND DEPTH, and the dispatch covers
/// the larger of the two block counts. A chain that has reached its output
/// passes a count of zero and is left alone by every level above it, so the
/// shorter chain is finished rather than padded and its scalar is written
/// exactly once. Every scalar is bit-identical to the two chains this merges:
/// each source is summed over the same [`FOLD_WIDTH`] in the same ascending
/// order at the same level.
///
/// THE MAGNITUDE LEVEL IS NOT THIS FUNCTION'S. Only the first level of an L1
/// norm takes absolute values, so the caller dispatches that one alone and
/// hands the result here as an ordinary array; see the call site.
///
/// # Safety
/// Every buffer must outlive the dispatches this appends, and both outputs must
/// name exactly one float. A finished chain names its own output as its source,
/// which is a real allocation and never [`ppf_cts_compute::Handle::NONE`]: a
/// generated entry resolves every handle it is given before the body runs.
unsafe fn encode_fold_dual(
    encoder: &mut dyn Encoder,
    first_source: ppf_cts_compute::Handle,
    first_count: usize,
    second_source: ppf_cts_compute::Handle,
    second_count: usize,
    scratch: &Buffer<f32>,
    first_base: usize,
    second_base: usize,
    first_out: ppf_cts_compute::Handle,
    second_out: ppf_cts_compute::Handle,
) -> Result<(), Fault> {
    debug_assert!(
        first_count > 0 && second_count > 0,
        "a fold of nothing has no level to dispatch"
    );
    let mut first_in = first_source;
    let mut second_in = second_source;
    let mut first_remaining = first_count;
    let mut second_remaining = second_count;
    // EACH CHAIN'S CURSOR STARTS PAST WHAT IS ALREADY WRITTEN, which is what
    // keeps a level from writing the scratch it is reading: a level reads
    // `[256 i, 256 i + 256)` and writes `i`, so an output overlapping its own
    // input is a race between threads rather than an aliasing question.
    let mut first_cursor = first_base;
    let mut second_cursor = second_base;
    let mut first_live = true;
    let mut second_live = true;
    loop {
        let first_blocks = if first_live {
            first_remaining.div_ceil(FOLD_WIDTH)
        } else {
            0
        };
        let second_blocks = if second_live {
            second_remaining.div_ceil(FOLD_WIDTH)
        } else {
            0
        };
        if first_blocks == 0 && second_blocks == 0 {
            return Ok(());
        }
        // A FINISHED CHAIN STILL NAMES REAL MEMORY. Its count is zero so no
        // thread writes for it, but the entry resolves both handles before the
        // body runs, so it points at its own output and reads one float it
        // already holds.
        let (first_src, first_dst, first_len) = if !first_live {
            (first_out, first_out, 1usize)
        } else if first_blocks == 1 {
            (first_in, first_out, first_remaining)
        } else {
            (
                first_in,
                scratch.span(first_cursor, first_blocks),
                first_remaining,
            )
        };
        let (second_src, second_dst, second_len) = if !second_live {
            (second_out, second_out, 1usize)
        } else if second_blocks == 1 {
            (second_in, second_out, second_remaining)
        } else {
            (
                second_in,
                scratch.span(second_cursor, second_blocks),
                second_remaining,
            )
        };
        let args = VecBlockSumDualCooperativeArgs {
            first_source: first_src,
            first_length: first_len as u32,
            second_source: second_src,
            second_length: second_len as u32,
            width: FOLD_WIDTH as u32,
            first_total: first_dst,
            second_total: second_dst,
            first_count: first_blocks as u32,
            second_count: second_blocks as u32,
            count: first_blocks.max(second_blocks) as u32,
            seam_arena_count: 0,
        };
        encoder.groups(
            &args,
            first_blocks.max(second_blocks) as u32,
            FOLD_LANES,
        )?;
        if first_blocks == 1 {
            first_live = false;
        } else if first_blocks > 1 {
            first_cursor += first_blocks;
            first_in = first_dst;
            first_remaining = first_blocks;
        }
        if second_blocks == 1 {
            second_live = false;
        } else if second_blocks > 1 {
            second_cursor += second_blocks;
            second_in = second_dst;
            second_remaining = second_blocks;
        }
    }
}

// Stage6: original pAp tree, with only its final writer-lane alpha fused.
unsafe fn encode_fold_alpha_stage6(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle, count: usize,
    scratch: &Buffer<f32>, pap: ppf_cts_compute::Handle,
    rz: ppf_cts_compute::Handle, alpha: ppf_cts_compute::Handle,
    status: ppf_cts_compute::Handle,
) -> Result<(), Fault> {
    let mut input = source;
    let mut remaining = count;
    let mut cursor = 0usize;
    loop {
        let blocks = remaining.div_ceil(FOLD_WIDTH);
        if blocks == 1 {
            let args = super::kernels::FemPcgReducedFoldAlphaLastArgs {
                source: input, length: remaining as u32, width: FOLD_WIDTH as u32,
                pap, rz, alpha, status, count: 1, seam_arena_count: 0,
            };
            return encoder.groups(&args, 1, FOLD_LANES);
        }
        let destination = scratch.span(cursor, blocks);
        let args = VecBlockSumCooperativeArgs {
            source: input, length: remaining as u32, width: FOLD_WIDTH as u32,
            total: destination, count: blocks as u32, seam_arena_count: 0,
        };
        encoder.groups(&args, blocks as u32, FOLD_LANES)?;
        cursor += blocks;
        input = destination;
        remaining = blocks;
    }
}

// Stage6: independent old L1/rz trees; terminal writer calls old stop.
unsafe fn encode_fold_dual_stop_stage6(
    encoder: &mut dyn Encoder,
    first_source: ppf_cts_compute::Handle,
    first_count: usize,
    second_source: ppf_cts_compute::Handle,
    second_count: usize,
    scratch: &Buffer<f32>,
    first_base: usize,
    second_base: usize,
    first_out: ppf_cts_compute::Handle,
    second_out: ppf_cts_compute::Handle,
    initial_error: f32, rz_previous: ppf_cts_compute::Handle,
    beta: ppf_cts_compute::Handle, relative: ppf_cts_compute::Handle,
    status: ppf_cts_compute::Handle,
    first_iteration: ppf_cts_compute::Handle,
    latched_abs_error: ppf_cts_compute::Handle,
    current_iteration: ppf_cts_compute::Handle,
    tolerance: f32, max_iterations: u32,
    first_already_done: bool,
) -> Result<(), Fault> {
    debug_assert!(
        first_count > 0 && second_count > 0,
        "a fold of nothing has no level to dispatch"
    );
    let mut first_in = first_source;
    let mut second_in = second_source;
    let mut first_remaining = first_count;
    let mut second_remaining = second_count;
    // EACH CHAIN'S CURSOR STARTS PAST WHAT IS ALREADY WRITTEN, which is what
    // keeps a level from writing the scratch it is reading: a level reads
    // `[256 i, 256 i + 256)` and writes `i`, so an output overlapping its own
    // input is a race between threads rather than an aliasing question.
    let mut first_cursor = first_base;
    let mut second_cursor = second_base;
    let mut first_live = !first_already_done;
    let mut second_live = true;
    loop {
        let first_blocks = if first_live {
            first_remaining.div_ceil(FOLD_WIDTH)
        } else {
            0
        };
        let second_blocks = if second_live {
            second_remaining.div_ceil(FOLD_WIDTH)
        } else {
            0
        };
        if first_blocks == 0 && second_blocks == 0 {
            return Ok(());
        }
        // A FINISHED CHAIN STILL NAMES REAL MEMORY. Its count is zero so no
        // thread writes for it, but the entry resolves both handles before the
        // body runs, so it points at its own output and reads one float it
        // already holds.
        let (first_src, first_dst, first_len) = if !first_live {
            (first_out, first_out, 1usize)
        } else if first_blocks == 1 {
            (first_in, first_out, first_remaining)
        } else {
            (
                first_in,
                scratch.span(first_cursor, first_blocks),
                first_remaining,
            )
        };
        let (second_src, second_dst, second_len) = if !second_live {
            (second_out, second_out, 1usize)
        } else if second_blocks == 1 {
            (second_in, second_out, second_remaining)
        } else {
            (
                second_in,
                scratch.span(second_cursor, second_blocks),
                second_remaining,
            )
        };
        if first_blocks <= 1 && second_blocks <= 1 {
            let args = super::kernels::FemPcgReducedFoldStopLastArgs {
                first_source: first_src, first_length: first_len as u32,
                second_source: second_src, second_length: second_len as u32,
                width: FOLD_WIDTH as u32,
                error: first_dst, rz_next: second_dst,
                first_count: first_blocks as u32,
                second_count: second_blocks as u32,
                initial_error, rz_previous, beta, relative, status,
                first_iteration, latched_abs_error, current_iteration,
                tolerance, max_iterations, count: 1, seam_arena_count: 0,
            };
            return encoder.groups(&args, 1, FOLD_LANES);
        }
        let args = VecBlockSumDualCooperativeArgs {
            first_source: first_src,
            first_length: first_len as u32,
            second_source: second_src,
            second_length: second_len as u32,
            width: FOLD_WIDTH as u32,
            first_total: first_dst,
            second_total: second_dst,
            first_count: first_blocks as u32,
            second_count: second_blocks as u32,
            count: first_blocks.max(second_blocks) as u32,
            seam_arena_count: 0,
        };
        encoder.groups(
            &args,
            first_blocks.max(second_blocks) as u32,
            FOLD_LANES,
        )?;
        if first_blocks == 1 {
            first_live = false;
        } else if first_blocks > 1 {
            first_cursor += first_blocks;
            first_in = first_dst;
            first_remaining = first_blocks;
        }
        if second_blocks == 1 {
            second_live = false;
        } else if second_blocks > 1 {
            second_cursor += second_blocks;
            second_in = second_dst;
            second_remaining = second_blocks;
        }
    }
}

/// The FIRST level of an L1 norm, dispatched alone so the levels above it can
/// share a chain with an unrelated fold.
///
/// Returns the array the rest of that chain folds and its length, which is the
/// source itself when one level was the whole fold.
///
/// # Safety
/// Every buffer must outlive the dispatch this appends.
unsafe fn encode_fold_abs_level(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    count: usize,
    scratch: &Buffer<f32>,
    out: ppf_cts_compute::Handle,
) -> Result<(ppf_cts_compute::Handle, usize), Fault> {
    let blocks = count.div_ceil(FOLD_WIDTH);
    let destination = if blocks == 1 { out } else { scratch.span(0, blocks) };
    let args = VecBlockSumAbsCooperativeArgs {
        source,
        length: count as u32,
        width: FOLD_WIDTH as u32,
        total: destination,
        count: blocks as u32,
        seam_arena_count: 0,
    };
    encoder.groups(
        &args,
        blocks as u32,
        FOLD_LANES,
    )?;
    Ok((destination, blocks))
}

/// `destination += sign * (*coefficient) * source`, with the coefficient read
/// from a device scalar rather than carried in the record.
///
/// # Safety
/// Every buffer must outlive the dispatch this appends.
unsafe fn encode_add_scaled_indirect(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    coefficient: ppf_cts_compute::Handle,
    sign: f32,
    n: usize,
) -> Result<(), Fault> {
    if n == 0 {
        return Ok(());
    }
    let args = VecAddScaledIndirectArgs {
        source,
        destination,
        coefficient,
        sign,
        count: n as u32,
        seam_arena_count: 0,
    };
    encoder.elements(&args, n as u32)
}

/// `p = z + beta p`, with `beta` read from a device scalar.
///
/// # Safety
/// Every buffer must outlive the dispatch this appends. `source_b` MAY alias
/// `destination`, which is the whole point of this shape: the body reads
/// element `i` before writing element `i` and every chunk is a disjoint span of
/// indices.
unsafe fn encode_combine_indirect(
    encoder: &mut dyn Encoder,
    source_a: ppf_cts_compute::Handle,
    source_b: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    a: f32,
    coefficient_b: ppf_cts_compute::Handle,
    n: usize,
) -> Result<(), Fault> {
    if n == 0 {
        return Ok(());
    }
    let args = VecCombineIndirectArgs {
        source_a,
        source_b,
        destination,
        a,
        coefficient_b,
        count: n as u32,
        seam_arena_count: 0,
    };
    encoder.elements(&args, n as u32)
}

/// `destination = a * source + b * destination`, the aliasing form.
///
/// # Safety
/// Both slices must outlive the dispatch this appends.
unsafe fn encode_combine_in_place(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    a: f32,
    b: f32,
    n: usize,
) -> Result<(), Fault> {
    encode_combine_raw(encoder, source, destination, destination, a, b, n)
}

/// Why a solve stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The relative residual reached the tolerance.
    Converged,
    /// Explicit opt-in inertia-scaled animation approximation.
    AcceptedAnimation,
    /// The iteration limit was reached first.
    MaxIterations,
    NoProgress,
    /// The curvature was unresolvable above the round-off of its own sum, so the
    /// iteration truncated and returned its iterate. Steihaug-style, and NOT an
    /// error: the direction so far is still a descent direction.
    CurvatureTruncated,
    /// The curvature was negative BEYOND the round-off bound. The assembled
    /// Newton Hessian is not SPD, which means a real assembly defect upstream:
    /// a per-element Hessian missing its PSD projection, a sign error, or a
    /// dropped off-diagonal block.
    ///
    /// SEPARATE FROM THE ONE BELOW BECAUSE THE REMEDIES DIFFER. This one sends
    /// a reader to the assembly; the other sends them to the preconditioner.
    /// One message for both would send every reader to the wrong half of the
    /// solver half the time.
    IndefiniteMatrix,
    /// `r^T M^-1 r` was non-positive. The PRECONDITIONER is not SPD.
    ///
    /// Every block-Jacobi diagonal block is inverted through a floored
    /// symmetric eigendecomposition, so each per-vertex term is positive by
    /// construction and their sum cannot be negative in exact arithmetic: a
    /// non-positive value here means a block is NaN or infinite, or the
    /// residual itself is not finite.
    NonSpdPreconditioner,
}

/// What a solve reports back.
#[derive(Debug, Clone, Copy)]
pub struct Report {
    pub outcome: Outcome,
    pub iterations: u32,
    /// Relative to the SEEDED initial residual, per the rule above.
    pub relative_residual: f32,
}

/// Build the block-Jacobi preconditioner into a caller-owned buffer.
///
/// Returns `false` when a block is unusable, which the caller must treat as a
/// fatal assembly defect rather than continuing: a zero block would not trip any
/// PCG guard and would silently freeze that vertex at its seed.
///
/// THE BUFFER IS THE CALLER'S BECAUSE THIS RUNS INSIDE A NEWTON ITERATION.
/// Allocating here would put a failure point in the middle of a solve, where
/// there is no good answer; the driver sizes it once at `initialize()`, where a
/// failure can be reported before a frame is written.
pub fn build_block_jacobi(diagonal: &[f32], inverse: &mut [f32], rows: u32) -> bool {
    block_jacobi_failure(diagonal, inverse, rows).is_none()
}

/// As [`build_block_jacobi`], naming the row it refused and what that row held.
///
/// THE ROW IS WHAT MAKES THE FAILURE ACTIONABLE. A refused block is an assembly
/// defect upstream rather than a tolerance, and the assembly that produced it is
/// identified by which vertex the row belongs to and what the nine floats are:
/// a NaN, an infinity and a negative eigenvalue are three different upstream
/// mistakes and a bare "not positive definite" separates none of them.
pub fn block_jacobi_failure(
    diagonal: &[f32],
    inverse: &mut [f32],
    rows: u32,
) -> Option<(usize, [f32; 9])> {
    assert_eq!(diagonal.len(), 9 * rows as usize, "one 3x3 block per row");
    assert_eq!(inverse.len(), 9 * rows as usize, "one 3x3 block per row");
    for row in 0..rows as usize {
        let mut lambda_max = 0.0f32;
        let ok = unsafe {
            block_jacobi_invert_abi(
                diagonal[9 * row..].as_ptr(),
                inverse[9 * row..].as_mut_ptr(),
                &mut lambda_max,
            )
        };
        if ok == 0 {
            let mut block = [0.0f32; 9];
            block.copy_from_slice(&diagonal[9 * row..9 * row + 9]);
            return Some((row, block));
        }
    }
    None
}

/// The vectors one PCG solve needs, owned by the caller for the same reason the
/// preconditioner's buffer is.
///
/// Sized once for the scene. `resize` here would be an allocation inside a
/// Newton iteration; the driver calls [`Workspace::size_for`] at
/// `initialize()`.
#[derive(Default)]
pub struct Workspace {
    ap: Buffer<f32>,
    /// THE RESIDUAL CARRIES A MIRROR, because the convergence test reads its
    /// L1 norm on the host once per iteration. That host dependency is not new:
    /// the residual was a host slice before and the same reduction ran on it
    /// every pass. What changed is that the transfer is now explicit.
    r: ReadbackBuffer<f32>,
    z: Buffer<f32>,
    p: Buffer<f32>,
    product: ReadbackBuffer<f32>,
    magnitude: ReadbackBuffer<f32>,
    /// Per row, the sum of the MAGNITUDES of that row's signed contributions to
    /// `p^T A p`, which the operator now forms while it multiplies.
    ///
    /// IT CANNOT SHARE `magnitude`. That one carries the per-row absolute
    /// product the dot terms yield, and both are live in the same region: the
    /// apply writes this one and the dot fill writes that one, an instruction
    /// apart.
    row_absolute: Buffer<f32>,
    /// Per row, the SIGNED sum of the same contributions `row_absolute` takes
    /// the magnitudes of, so `p^T A p` and its round-off bound come from ONE
    /// accumulation. Forming the curvature separately as `p . Ap` after the
    /// product would be a different rounding on the one scalar the step length
    /// divides by, and the bound would then be measuring a sum that is not the
    /// one being bounded.
    row_curvature: Buffer<f32>,
    /// The alpha / beta verdict: the scalar and the noise floor beside it.
    ///
    /// A WORKSPACE FIELD RATHER THAN A STACK SCALAR, because these run once per
    /// PCG iteration and a per-call device allocation would be paid every one of
    /// them. `Buffer::size` grows only past capacity, so the sizing below costs
    /// nothing after the first.
    verdict: ReadbackBuffer<f32>,
    /// The body's own classification of that scalar, which is an `i32` and so
    /// cannot share the buffer above.
    verdict_cause: ReadbackBuffer<i32>,
    /// EVERY SCALAR THE RECURRENCE CARRIES, in one allocation, laid out by
    /// [`slot`].
    ///
    /// It is a `ReadbackBuffer` for one field only: `||r||_1` is what the
    /// convergence test reads, and the loop cannot decide to stop without it.
    /// The other nine are written by one dispatch and read by the next without
    /// ever crossing the seam.
    scalar: ReadbackBuffer<f32>,
    /// The two verdicts' causes, in one allocation laid out by [`cause_slot`].
    cause: ReadbackBuffer<i32>,
    /// The fold's intermediate levels.
    ///
    /// A PLAIN `Buffer`: no level of a fold is ever read by the host, only the
    /// last one's destination, which is a slot of `scalar` above.
    fold: Buffer<f32>,
}

impl Workspace {
    /// Size every vector for a scene of `rows` vertices.
    pub fn size_for(&mut self, device: &mut impl Device, rows: u32) -> Result<(), Fault> {
        let n = 3 * rows as usize;
        self.ap.size(device, n, AllocLabel("pcg.ap"))?;
        self.r.size(device, n, AllocLabel("pcg.r"))?;
        self.z.size(device, n, AllocLabel("pcg.z"))?;
        self.p.size(device, n, AllocLabel("pcg.p"))?;
        // THE TWO FOLD OUTPUTS ARE DEVICE ALLOCATIONS, written by the per-row
        // pass and reduced on the host, so they carry a mirror and their sizing
        // needs a backend where the four vectors above do not.
        self.product
            .size(device, rows as usize, AllocLabel("pcg.product"))?;
        self.verdict
            .size(device, 2, AllocLabel("pcg.verdict"))?;
        self.verdict_cause
            .size(device, 1, AllocLabel("pcg.verdict_cause"))?;
        self.magnitude
            .size(device, rows as usize, AllocLabel("pcg.magnitude"))?;
        self.row_absolute
            .size(device, rows as usize, AllocLabel("pcg.row_absolute"))?;
        self.row_curvature
            .size(device, rows as usize, AllocLabel("pcg.row_curvature"))?;
        // THE RECURRENCE'S SCALARS AND THE FOLD THAT PRODUCES THEM. Sized once
        // for the scene like everything else here: a fold level's extent is
        // decided by the vector it reduces, so the deepest chain any call makes
        // is the one over `n` floats and the scratch is sized for that.
        self.scalar
            .size(device, slot::COUNT, AllocLabel("pcg.scalar"))?;
        self.cause
            .size(device, cause_slot::COUNT, AllocLabel("pcg.cause"))?;
        // TWICE THE CHAIN, because `encode_fold_pair` runs two arrays through
        // the levels at once and a level reads both while writing both, so one
        // region cannot serve. The single-array folds beside it use the first
        // half and are unaffected.
        self.fold.size(
            device,
            (2 * fold_scratch_len(n)).max(1),
            AllocLabel("pcg.fold"),
        )?;
        Ok(())
    }
}

/// `z = M^-1 r`, one 3x3 block per row.
///
/// # Safety
/// Every slice must outlive the dispatch this appends.
unsafe fn encode_preconditioner(
    encoder: &mut dyn Encoder,
    inverse: ppf_cts_compute::Handle,
    r: ppf_cts_compute::Handle,
    z: ppf_cts_compute::Handle,
    rows: u32,
) -> Result<(), Fault> {
    if rows == 0 {
        return Ok(());
    }
    // The record is the BODY's, not this pass's: `mat3_mul` takes a 3x3
    // block and a vector and returns the product, and one dispatch of it per
    // row IS the block-diagonal apply. The entry point reads the block at a
    // stride of nine floats and the vector at three, which is the addressing
    // a hand-written range shim would have to spell out.
    let args = Mat3MulArgs {
        matrix: inverse,
        vector: r,
        result: z,
        count: rows,
        seam_arena_count: 0,
    };
    encoder.elements(&args, rows)
}

/// The relative residual, refreshed from the device first.
///
/// `handle()` stales the mirror, so a report built after a dispatch has to read
/// the array again rather than answer out of the previous pass.
fn residual_of<D: Device>(
    device: &mut D,
    r: &mut ReadbackBuffer<f32>,
    err0: f32,
) -> Result<f32, Fault> {
    r.download(device)?;
    Ok(reduce::sum_abs(r.host()) / err0)
}

/// Solve `M x = b` by preconditioned conjugate gradient, with the recurrence
/// on the device.
///
/// `x` enters carrying the caller's initial guess and leaves carrying the
/// iterate. That is not an optimization: the driver seeds prescribed rows with
/// their exact correction, and the denominator rule above depends on the
/// residual being formed against that seed.
///
/// # What crosses the seam, and what does not
///
/// EVERY PER-ITERATION SCALAR STAYS ON THE DEVICE, and an iteration is issued
/// as one unbroken stream of kernels; the host reads a small probe and nothing
/// else. One region per iteration carries the matvec, the three folds the
/// curvature verdict needs, that verdict, the two updates, the residual norm,
/// the preconditioner, `r . z`, the beta verdict with its roll, and the
/// direction update; then the host reads the two scalar blocks, which are ten
/// floats and two ints.
///
/// THE ALTERNATIVE IS FOUR FULL-LENGTH DOWNLOADS AN ITERATION, one of the
/// residual and three of the per-row product array, each folded on the host. A
/// readback of a scalar is TRANSPORT; downloading an array and folding it is the
/// computation itself moved to the host, which rule (1a-0) forbids because it
/// makes this a different solver rather than a slower one.
///
/// # Two things this deliberately does NOT change
///
/// **The termination ORDER is fixed**: the alpha verdict is tested first, then
/// convergence, then the beta verdict. Reordering the three changes which
/// iterate a marginal solve returns, so it is a decision of its own and never a
/// side effect of a change about where arithmetic runs.
///
/// **The iteration is RECORDED ONCE AND REPLAYED**: the loop encodes one
/// iteration and replays it, sampling the residual one time in four while the
/// residual is far from the tolerance and every iteration once within
/// `NEAR_TOL_FACTOR` of it. The batch size and the sampling rate are ONE number,
/// because a backend synchronizes at the end of every submit, so a check the
/// batch does not end at buys nothing.
#[allow(clippy::too_many_arguments)]
pub fn solve<D: Device>(
    device: &mut D,
    op: &Operator,
    inverse_diagonal: ppf_cts_compute::Handle,
    b: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    work: &mut Workspace,
    // THE PRECONDITIONER, when it is not the block-diagonal one.
    //
    // `None` is block-Jacobi, the default, and stays the fast path: one
    // dispatch of `mat3_mul` per row IS the block-diagonal apply. `Some` is an
    // additive Schwarz sweep, three dispatches over the domains the caller
    // built. Both are pushed onto the same encoder, so an iteration still
    // costs one submission either way.
    mut schwarz: Option<&mut super::schwarz::State>,
    tolerance: f32,
    max_iterations: u32,
) -> Result<Report, Fault> {
    let rows = op.rows();
    let n = 3 * rows as usize;
    assert_eq!(work.ap.len(), n, "the workspace was sized for a different scene");

    let Workspace {
        ap,
        r,
        z,
        p,
        product,
        magnitude,
        row_absolute,
        row_curvature,
        scalar,
        cause,
        fold,
        ..
    } = work;
    let mut terms = DotTerms {
        product,
        magnitude,
    };
    // A SCENE WITH NO ROWS HAS NOTHING TO FOLD, and a fold of nothing has no
    // level to dispatch. It is already solved by inspection.
    if n == 0 {
        return Ok(Report {
            outcome: Outcome::Converged,
            iterations: 0,
            relative_residual: 0.0,
        });
    }
    let term_rows = terms.rows();

    // r = b - A x0, the preconditioner, the first search direction and BOTH
    // opening scalars, in ONE boundary. Consecutive entries of a region are
    // ordered with a full barrier between them on every backend, so a phase
    // whose steps feed each other costs one host round trip rather than one per
    // dispatch.
    {
        let ap_h = ap.handle();
        let r_h = r.handle();
        let z_h = z.handle();
        let p_h = p.handle();
        let product_h = terms.product_handle();
        let err_out = scalar.span(slot::ERR, 1);
        let rz0_out = scalar.span(slot::RZ0, 1);
        // CLEAR THE SCALAR BUFFER BEFORE THE PHASE THAT FILLS IT, which the
        // announcement flag needs and nothing else here minds: every other slot
        // is written by a dispatch below or by the loop before anything reads
        // it. The flag is sticky within a solve and the workspace is reused by
        // every solve, so without this a breakdown in one solve would send the
        // next one to read a latch that the `cause` seed had already cleared,
        // and a Metal arena hands out allocations without zeroing besides.
        //
        // A WHOLE-BUFFER SEED RATHER THAN A ONE-SLOT FILL, because a fill takes
        // a live allocation and a slot is a span inside one.
        scalar.seed(device, &[0.0; slot::COUNT])?;
        device.run("pcg.seed", |encoder| {
            // Safety: every buffer is borrowed for the whole call, and a
            // sweep's handles outlive it for the same reason.
            unsafe {
                op.encode_apply(encoder, x, ap_h, row_absolute.handle(),
                                row_curvature.handle())?;
                encode_combine_raw(encoder, b, ap_h, r_h, 1.0, -1.0, n)?;
                // err0 = ||r||_1, the denominator every relative residual below
                // is measured against. It is the SEEDED residual and never
                // ||b||_1; see the module doc.
                encode_fold(encoder, r_h, n, fold, err_out, true)?;
                match schwarz.as_deref_mut() {
                    // THE WHOLE SWEEP, fine level and every coarse level,
                    // pushed onto this one encoder so an iteration still costs
                    // one submission whichever preconditioner runs.
                    Some(state) => super::schwarz::encode_whole_sweep(
                        encoder, state, r_h, z_h, rows)?,
                    None => encode_preconditioner(
                        encoder, inverse_diagonal, r_h, z_h, rows)?,
                }
                // p = z, which is the beta-free case of the combine the loop
                // uses below.
                encode_combine_raw(encoder, z_h, p_h, p_h, 1.0, 0.0, n)?;
                terms.encode_fill(encoder, r_h, z_h, n)?;
                encode_fold(encoder, product_h, term_rows, fold, rz0_out, false)
            }
        })?;
    }
    // THE ONE HOST READ AT SETUP. It answers two questions at once: the
    // trivially-already-converged case, and whether the preconditioner is SPD
    // at the seed.
    scalar.download(device)?;
    let err0 = scalar.host()[slot::ERR];
    if err0 == 0.0 {
        return Ok(Report {
            outcome: Outcome::Converged,
            iterations: 0,
            relative_residual: 0.0,
        });
    }
    let rz0 = scalar.host()[slot::RZ0];
    // The negation is deliberate and clippy's suggestion would break it: NaN
    // fails `> 0.0`, so `!(rz0 > 0.0)` traps a NaN where `rz0 <= 0.0` would let
    // it through and let the solve run on garbage.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(rz0 > 0.0) {
        // The preconditioner is SPD by construction, so a non-positive r.z
        // means the residual or the preconditioner is broken upstream.
        return Ok(Report {
            outcome: Outcome::NonSpdPreconditioner,
            iterations: 0,
            relative_residual: 1.0,
        });
    }

    // The last residual the loop measured, which the iteration-cap exit reports.
    // Re-folding `r` at that exit would read the same residual out of the same
    // unchanged vector, so the loop's last measurement is carried instead.
    let mut residual = 1.0f32;
    // THE BATCH SIZE, WHICH IS ALSO THE RESIDUAL SAMPLING SCHEDULE.
    //
    // Every per-iteration scalar already stays on the device: the recurrence is
    // folds and indirect axpys, and alpha and beta are decided by one-element
    // resident bodies. The only thing crossing the seam is this probe, and
    // crossing on EVERY iteration is a full pipeline stall per iteration.
    // It is read on a stride of 4 while the residual is far from the tolerance
    // and every iteration once within 8x of it, so a sub-tol crossing is never
    // sampled past and convergence is detected at the same iterate an unstrided
    // loop would have found.
    //
    // WHAT MAKES IT SAFE IS THE LATCH, not the schedule. A breakdown between
    // two checks is held in the three sticky slots the resident bodies write,
    // so it is reported at the next check with the offending iterate's own
    // values. Without that latch this would silently drop a `pAp <= 0`.
    const RESID_CHECK_STRIDE: u32 = 4;
    const NEAR_TOL_FACTOR: f32 = 8.0;
    let mut stride = RESID_CHECK_STRIDE;
    // CLEAR THE LATCH BEFORE THE LOOP. The workspace is allocated once and
    // reused by every solve, and the latch is sticky BY DESIGN, so without this
    // a breakdown in one solve would abort the next one before it had taken a
    // single iteration. A Metal arena also hands out allocations without
    // zeroing and never faults on an uninitialized read, so the first solve
    // needs it as much as the second.
    cause.seed(device, &[PCG_BREAK_NONE; cause_slot::COUNT])?;
    // ONE SUBMIT PER BATCH, NOT PER ITERATION, and that is the whole point.
    //
    // A BACKEND SYNCHRONIZES AT THE END OF EVERY SUBMIT, which is the contract
    // `Device::run` is defined under, so a submit per iteration is a full
    // device stall per iteration whatever the probe does. Striding only the READ
    // leaves the stall in place and was measured at 1.24x; aligning the SUBMIT
    // with the check is the change that matters.
    //
    // Nothing on the host is needed between iterations: every per-iteration
    // scalar is device-resident and the breakdown is held in the sticky latch,
    // so `stride` iterations encode back to back into one batch. The stride is
    // therefore the BATCH SIZE and the sampling rate at once.
    // THE FIRST CHECK IS AT ITERATION ONE, so the
    // first iteration is sampled on its own and the residual has a chance to
    // tighten the stride before any coarse batch runs. Skipping that first
    // sample quantizes a short solve's reported iteration count to a multiple
    // of the stride, which is enough to hide a preconditioner that is not being
    // applied.
    // THE RECORDED ITERATION, captured once after a warm-up iteration and
    // replayed for every batch after the first, which collapses the iteration's
    // launches into one graph launch.
    //
    // IT IS RECORDABLE ONLY BECAUSE NOTHING IN THE BODY DEPENDS ON WHICH TURN
    // OF THE LOOP IT IS. Every per-iteration scalar is device-resident, and the
    // last host-supplied one, the iteration index, is a device counter the
    // alpha pass advances for itself.
    //
    // AND ONLY BECAUSE THE FAILURE MODE IS LATCH AND CONTINUE, which is
    // `Device::replay`'s stated precondition: a breakdown on repeat `k` is
    // reported after the whole batch has run, and the sticky `break_cause`
    // latch is exactly what makes that safe. The strided probe already accepted
    // that delay.
    let mut region: Option<D::Region> = None;
    // THE VERDICT, CARRIED OUT OF THE LOOP RATHER THAN RETURNED FROM INSIDE IT,
    // so the recorded region is released on every path that reaches an answer.
    // `Device::Region` has no `Drop`: `release` takes it by value and must be
    // called, and this solve runs thousands of times a frame.
    let mut finished: Option<Report> = None;
    let mut next_check = 1u32;
    let mut iteration = 0u32;
    while iteration < max_iterations {
        let target = next_check.min(max_iterations);
        let batch = target - iteration;
        let _first = iteration + 1;
        iteration = target;
        let ap_h = ap.handle();
        let r_h = r.handle();
        let z_h = z.handle();
        let p_h = p.handle();
        let product_h = terms.product_handle();
        let rz0_slot = scalar.span(slot::RZ0, 1);
        let rz1_slot = scalar.span(slot::RZ1, 1);
        let p_ap_slot = scalar.span(slot::P_AP, 1);
        let alpha_slot = scalar.span(slot::ALPHA, 1);
        let beta_slot = scalar.span(slot::BETA, 1);
        let err_slot = scalar.span(slot::ERR, 1);
        // ONE SLOT NOW, NOT TWO. The scale is the magnitude sum the apply
        // forms; the two Cauchy-Schwarz halves it replaced are gone.
        let absdot_slot = scalar.span(slot::ABSDOT, 1);
        // The three STICKY latch slots. They are written only on the first
        // breakdown and read at a scheduled check, which is what lets the host
        // sample on a stride without losing one.
        let break_value_slot = scalar.span(slot::BREAK_VALUE, 1);
        let break_cause_slot = cause.span(cause_slot::BREAK_CAUSE, 1);
        let break_iter_slot = cause.span(cause_slot::BREAK_ITER, 1);
        let iteration_slot = cause.span(cause_slot::ITERATION, 1);
        let break_fired_slot = scalar.span(slot::BREAK_FIRED, 1);
        let absolute_h = row_absolute.handle();
        // ONE GROUP PER `APPLY_GROUP` ROWS, the same split the folded apply
        // uses, so both per-group partial arrays are indexed the same way and
        // both are at most `rows / APPLY_GROUP` long.
        let update_groups = rows.div_ceil(super::operator::APPLY_GROUP);
        let curvature_h = row_curvature.handle();
        let alpha_noise_slot = scalar.span(slot::ALPHA_NOISE, 1);
        let beta_noise_slot = scalar.span(slot::BETA_NOISE, 1);
        let alpha_cause_slot = cause.span(cause_slot::ALPHA, 1);
        let beta_cause_slot = cause.span(cause_slot::BETA, 1);

        if region.is_none() {
            region = Some(device.record("pcg.iteration", |encoder| {
            // Safety: every buffer is borrowed for the whole call, consecutive
            // entries of a region are ordered, and the only deliberate aliasing
            // is `p` in the direction update, whose body reads element `i`
            // before writing element `i`.
            unsafe {
                // THE APPLY FOLDS ITS OWN TWO SCALARS, per group rather
                // than per row, so what follows reduces `groups` floats and
                // not `rows` of them. Measured before this: 81,920 rows folded
                // 320, then 2, then 1, three dispatches for each of two chains
                // every iteration.
                let groups =
                    op.encode_apply_folded(encoder, p_h, ap_h, curvature_h, absolute_h)?;
                // p^T A p, and the scale its round-off bound is taken from.
                //
                // THE SCALE COMES OUT OF THE APPLY ITSELF: the operator sums
                // the MAGNITUDES of the signed contributions it is already
                // computing, per row, and this folds them. That is one fold
                // rather than the two the
                // Cauchy-Schwarz surrogate needed, and two fewer full-length
                // fills, because `|p|` and `|Ap|` are not wanted at all.
                // BOTH SCALARS COME OUT OF THE APPLY, from one pass over the
                // same terms. A separate dot fill would walk the product a
                // second time and give the curvature a different rounding from
                // the bound measuring it.
                // ONE CHAIN FOR BOTH. The two arrays have the same length and
                // the same shape, so a level folds both in one dispatch.
                // THE FOLD TAKES THE VERDICT, so no dispatch of its own stands
                // between the sums and the alpha they decide. The
                // classification, the sticky latch and the announcement are all
                // `pcg_alpha_resident`'s, reached from inside the fold's last
                // level.
                let alpha_fold_args = PcgFoldAlphaArgs {
                    curvature_source: curvature_h,
                    absolute_source: absolute_h,
                    length: groups,
                    curvature_out: p_ap_slot,
                    absolute_out: absdot_slot,
                    rz: rz0_slot,
                    iteration_counter: iteration_slot,
                    value_out: alpha_slot,
                    noise_out: alpha_noise_slot,
                    cause_out: alpha_cause_slot,
                    break_cause: break_cause_slot,
                    break_value: break_value_slot,
                    break_iteration: break_iter_slot,
                    break_fired: break_fired_slot,
                    count: 1,
                    seam_arena_count: 0,
                };
                encoder.groups(&alpha_fold_args, 1, FOLD_LANES)?;
                // x += alpha p and r -= alpha Ap. A breakdown leaves `alpha` at
                // zero, so both updates are exact no-ops and the iterate the
                // report returns is the one the guard fired at.
                match schwarz.as_deref_mut() {
                    // THE SCHWARZ PATH STAYS SPLIT. Its sweep is a multi-level
                    // pass of its own shape, not a per-row apply, so there is
                    // nothing for the fused row to absorb and the four
                    // dispatches below are what it costs.
                    Some(state) => {
                        encode_add_scaled_indirect(encoder, p_h, x, alpha_slot, 1.0, n)?;
                        encode_add_scaled_indirect(
                            encoder, ap_h, r_h, alpha_slot, -1.0, n)?;
                        super::schwarz::encode_whole_sweep(
                            encoder, state, r_h, z_h, rows)?;
                        terms.encode_fill(encoder, r_h, z_h, n)?;
                    }
                    // ONE PASS OVER THE ROW for the iterate, the residual, the
                    // preconditioned residual and the `r . z` terms. A breakdown
                    // leaves `alpha` at zero, so the two updates are exact
                    // no-ops and the iterate the report returns is the one the
                    // guard fired at.
                    None => {
                        // THE ROW FOLDS ITS OWN TWO SCALARS, per group. The
                        // element form left `r . z` per ROW and the residual
                        // norm to a magnitude level over `3 * rows` floats, so
                        // what followed was an abs level plus a three-level
                        // dual chain. Both sums are per group here, so they
                        // share one length and one dispatch folds them.
                        let update_args = PcgUpdateRowFoldedArgs {
                            direction: p_h,
                            product_direction: ap_h,
                            alpha: alpha_slot,
                            inverse_diagonal,
                            iterate: x,
                            residual: r_h,
                            preconditioned: z_h,
                            product_total: terms.product_handle(),
                            residual_total: terms.magnitude_handle(),
                            rows,
                            count: update_groups,
                            seam_arena_count: 0,
                        };
                        encoder.groups(&update_args, update_groups, super::operator::APPLY_GROUP)?;
                    }
                }
                // ||r||_1 at the new iterate. Folding it BEFORE the
                // preconditioner would give the same number, since the
                // preconditioner only reads `r`. The fused row writes `r` and
                // `z` in one pass, so there is no point before it to fold at,
                // and the value is the same either way.
                // ONE CHAIN FOR BOTH, which is a launch-count change and
                // nothing else. The two folds are independent and adjacent
                // with nothing between them, so every level of one can share a
                // dispatch with a level of the other; they differ only in
                // length, which `encode_fold_dual` carries per source. Each
                // scalar is summed over the same width in the same ascending
                // order at the same level as the chain it replaces.
                //
                // THE MAGNITUDE LEVEL GOES FIRST AND ALONE. Only the first
                // level of an L1 norm takes absolute values, so it is its own
                // dispatch and the merge begins above it. When that one level
                // is the whole residual fold there is nothing left to merge
                // and `r . z` folds by itself.
                match schwarz.as_deref_mut() {
                    // THE SCHWARZ PATH KEEPS THE CHAIN, because its sweep is a
                    // pass of its own shape and its row is not the fused one:
                    // `terms.encode_fill` still writes per-ROW arrays there.
                    Some(_) => {
                        let (residual_source, residual_count) =
                            encode_fold_abs_level(encoder, r_h, n, fold, err_slot)?;
                        if residual_count == 1 {
                            encode_fold(
                                encoder, product_h, term_rows, fold, rz1_slot, false)?;
                        } else {
                            encode_fold_dual(
                                encoder,
                                residual_source,
                                residual_count,
                                product_h,
                                term_rows,
                                fold,
                                residual_count,
                                fold_scratch_len(n),
                                err_slot,
                                rz1_slot,
                            )?;
                        }
                        // BETA STAYS ITS OWN DISPATCH ON THIS PATH, because the
                        // chain above ends in a plain fold that takes no
                        // verdict. The block-Jacobi arm below folds and
                        // classifies in one group.
                        let beta_args = PcgBetaResidentArgs {
                            rz_next: rz1_slot,
                            rz_previous: rz0_slot,
                            iteration_counter: iteration_slot,
                            value_out: beta_slot,
                            noise_out: beta_noise_slot,
                            cause_out: beta_cause_slot,
                            break_cause: break_cause_slot,
                            break_value: break_value_slot,
                            break_iteration: break_iter_slot,
                            break_fired: break_fired_slot,
                            count: 1,
                            seam_arena_count: 0,
                        };
                        encoder.elements(&beta_args, 1)?;
                    }
                    // Both partials are one float per group and share a length,
                    // so this is the same single dispatch the apply's pair uses.
                    None => {
                        // The same fold-and-verdict as alpha's, one step later:
                        // beta, the roll of `r . z` and the non-positive test
                        // all happen inside the fold that finishes the sums they
                        // read.
                        let beta_fold_args = PcgFoldBetaArgs {
                            product_source: product_h,
                            residual_source: terms.magnitude_handle(),
                            length: update_groups,
                            product_out: rz1_slot,
                            residual_out: err_slot,
                            rz_previous: rz0_slot,
                            iteration_counter: iteration_slot,
                            value_out: beta_slot,
                            noise_out: beta_noise_slot,
                            cause_out: beta_cause_slot,
                            break_cause: break_cause_slot,
                            break_value: break_value_slot,
                            break_iteration: break_iter_slot,
                            break_fired: break_fired_slot,
                            count: 1,
                            seam_arena_count: 0,
                        };
                        encoder.groups(&beta_fold_args, 1, FOLD_LANES)?;
                    }
                }
                encode_combine_indirect(encoder, z_h, p_h, p_h, 1.0, beta_slot, n)?;
            }
            Ok(())
            })?);
        }
        // THE ONE SUBMIT THE BATCH PAYS, whatever its size. The region is
        // replayed back to back with no host round trip inside it, and the
        // diagnostic channel is collected once for the whole batch.
        device.replay(
            region.as_ref().expect("the region was recorded on this pass"),
            batch,
        )?;

        // THE PROBE, once per BATCH. The batch is sized so it ends on the
        // scheduled check, so the submit's own synchronize is the only one the
        // iteration pays and this readback rides it rather than adding a
        // second.
        scalar.download(device)?;
        residual = scalar.host()[slot::ERR] / err0;
        // THE INT LATCH IS READ ONLY WHEN IT HOLDS SOMETHING. Both guards
        // announce a breakdown by writing `BREAK_FIRED` into the scalar buffer
        // this probe already downloads, so a healthy batch costs ONE
        // device-to-host copy rather than two. Measured on `drape`,
        // `pcg.cause` was 939 of 2,404 such copies.
        //
        // THE FLAG IS NECESSARY AND NOT MERELY SUFFICIENT, which is what makes
        // skipping the read safe: it is written by the same branch that writes
        // `break_cause`, so the int slot cannot hold a cause while the flag
        // reads zero. It is deliberately NOT derived from `BREAK_VALUE`, which
        // a real `pAp <= 0` breakdown can legitimately latch as 0.0.
        let (alpha_cause, break_iteration) = if scalar.host()[slot::BREAK_FIRED] != 0.0 {
            cause.download(device)?;
            // THE LATCHED CAUSE, not this iteration's. Between two checks the
            // live `cause_out` slots have been rewritten up to three times; the
            // sticky slot holds whichever breakdown fired first and the iterate
            // it fired at.
            (
                cause.host()[cause_slot::BREAK_CAUSE],
                cause.host()[cause_slot::BREAK_ITER],
            )
        } else {
            (PCG_BREAK_NONE, 0)
        };
        // CONVERGENCE IS TESTED FIRST, and the latch makes that REQUIRED rather
        // than a preference. A sticky flag can have fired several iterations
        // ago while the solve went on to converge; a finite residual below tol
        // wins in that case, and a NaN or Inf residual fails this test and
        // falls through to the breakdown branches below.
        let mut true_residual_rejected = false;
        if residual < tolerance {
            // This entry point returns x unchanged, so probe exactly that x.
            // The original raw seeded L1 denominator err0 remains fixed.
            let recursive_residual = residual;
            let probe = (|| -> Result<f32, Fault> {
                let err_out = scalar.span(slot::ERR, 1);
                device.run("pcg.true_residual", |encoder| {
                    unsafe {
                        op.encode_apply(encoder, x, ap_h, absolute_h, curvature_h)?;
                        encode_combine_raw(encoder, b, ap_h, r_h, 1.0, -1.0, n)?;
                        encode_fold(encoder, r_h, n, fold, err_out, true)
                    }
                })?;
                scalar.download(device)?;
                Ok(scalar.host()[slot::ERR] / err0)
            })();
            residual = match probe {
                Ok(value) => value,
                Err(error) => {
                    if let Some(recorded) = region.take() { device.release(recorded); }
                    return Err(error);
                }
            };
            if residual < tolerance {
                finished = Some(Report {
                    outcome: Outcome::Converged,
                    iterations: iteration,
                    relative_residual: residual,
                });
                break;
            }
            ::log::info!("pcg true-residual rejected at iteration {}: recursive={:.9} true={:.9}",
                iteration,recursive_residual,residual);
            true_residual_rejected = true;
            // A true-residual rejection does not erase a previously latched
            // curvature/SPD verdict. Its original precedence is handled below.
        }
        // THE LATCHED CAUSE IS MATCHED EXHAUSTIVELY, because one slot now
        // carries all three. Alpha latches `PCG_BREAK_PAP` and
        // `PCG_BREAK_NOISE`, beta latches `PCG_BREAK_RZ`, and reading anything
        // that is not `PAP` as a truncation, which is what a two-way test did
        // while the slots were separate, would report a non-SPD preconditioner
        // as a benign Steihaug stop.
        if alpha_cause != PCG_BREAK_NONE {
            // The iterate the latch fired at, not the check that noticed it.
            let broke_at = if break_iteration > 0 {
                break_iteration as u32
            } else {
                iteration
            };
            let outcome = match alpha_cause {
                // Negative beyond the bound, or NaN: the sign is real, so the
                // assembled Newton Hessian is not SPD.
                PCG_BREAK_PAP => Outcome::IndefiniteMatrix,
                // The previous iterate's `r^T M^-1 r` was non-positive, so beta
                // could not be formed.
                PCG_BREAK_RZ => Outcome::NonSpdPreconditioner,
                // Unresolvable above the round-off of its own sum. Steihaug:
                // the direction so far is still a descent direction.
                //
                // REPORTED HERE rather than by the caller. A truncated
                // solve returns SUCCESS and an iterate the Newton line search
                // then accepts, so without this line a step that stopped early
                // on an unresolvable curvature is indistinguishable in the log
                // from one that converged.
                _ => {
                    ::log::info!(
                        "* cg truncated: curvature {:.3e} within round-off of its own \
                         reduction at iter {} (reresid {:.3e})",
                        scalar.host()[slot::BREAK_VALUE],
                        broke_at,
                        residual
                    );
                    Outcome::CurvatureTruncated
                }
            };
            finished = Some(Report {
                outcome,
                iterations: broke_at,
                relative_residual: residual,
            });
            break;
        }
        if true_residual_rejected {
            if iteration >= max_iterations || !residual.is_finite() {
                finished = Some(Report {
                    outcome: Outcome::MaxIterations,
                    iterations: iteration,
                    relative_residual: residual,
                });
                break;
            }
            // No old breakdown survived the precedence check. Restart with
            // true r and the SAME preconditioner, preserving the resident
            // iteration counter, original err0 and recorded graph handles.
            let restart = (|| -> Result<f32, Fault> {
                device.run("pcg.true_residual_restart", |encoder| {
                    unsafe {
                        match schwarz.as_deref_mut() {
                            Some(state) => super::schwarz::encode_whole_sweep(
                                encoder, state, r_h, z_h, rows)?,
                            None => encode_preconditioner(
                                encoder, inverse_diagonal, r_h, z_h, rows)?,
                        }
                        encode_combine_raw(encoder, z_h, p_h, p_h, 1.0, 0.0, n)?;
                        terms.encode_fill(encoder, r_h, z_h, n)?;
                        encode_fold(encoder, product_h, term_rows, fold, rz0_slot, false)
                    }
                })?;
                scalar.download(device)?;
                Ok(scalar.host()[slot::RZ0])
            })();
            let restarted_rz = match restart {
                Ok(value) => value,
                Err(error) => {
                    if let Some(recorded) = region.take() { device.release(recorded); }
                    return Err(error);
                }
            };
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(restarted_rz > 0.0) {
                finished = Some(Report {
                    outcome: Outcome::NonSpdPreconditioner,
                    iterations: iteration,
                    relative_residual: residual,
                });
                break;
            }
            stride = 1;
            next_check = iteration + 1;
            continue;
        }
        // Within reach of the tolerance: check every iteration from here so the
        // crossing is detected at the same iterate the unstrided loop would
        // have found.
        if residual < NEAR_TOL_FACTOR * tolerance {
            stride = 1;
        }
        next_check = iteration + stride;
    }

    if let Some(region) = region {
        device.release(region);
    }
    Ok(finished.unwrap_or(Report {
        outcome: Outcome::MaxIterations,
        iterations: max_iterations,
        relative_residual: residual,
    }))
}

/// One scalar from `pcg_alpha` / `pcg_beta`, with the body's own
/// classification of it.
struct PcgScalar {
    value: f32,
    cause: i32,
}

fn pcg_alpha<D: Device>(
    device: &mut D,
    // THE TWO OUT-BUFFERS, not the whole workspace: the caller has already
    // destructured it into disjoint field borrows, so a second borrow of the
    // whole would conflict with the vectors the same iteration is using.
    verdict: &mut ReadbackBuffer<f32>,
    verdict_cause: &mut ReadbackBuffer<i32>,
    rz: f32,
    p_ap: f32,
    absolute_dot: f32,
) -> Result<PcgScalar, Fault> {
    let args = PcgAlphaTermsArgs {
        // THE THREE OUT-PARAMETERS ARE THE WORKSPACE'S, not stack scalars: a
        // handle names an allocation, and a local has none.
        value_out: verdict.span(0, 1),
        noise_out: verdict.span(1, 1),
        cause_out: verdict_cause.span(0, 1),
        rz,
        p_ap,
        absolute_dot,
        count: 1,
        seam_arena_count: 0,
    };
    // ONE ELEMENT, because that is what it is: a scalar verdict evaluated by the
    // shared body. The device-resident loop reaches the same body from inside
    // its fold instead, which is a dispatch shape rather than different
    // arithmetic.
    // Safety: the three out-parameters outlive the dispatch.
    unsafe { device.launch("pcg.alpha", &args, 1) }?;
    // THE MIRRORS, brought current before the verdict is read: taking a handle
    // above staled both by construction.
    verdict.download(device)?;
    verdict_cause.download(device)?;
    let value = verdict.host()[0];
    let noise = verdict.host()[1];
    let cause = verdict_cause.host()[0];
    let _ = noise;
    Ok(PcgScalar { value, cause })
}

fn pcg_beta<D: Device>(
    device: &mut D,
    // THE TWO OUT-BUFFERS, not the whole workspace: the caller has already
    // destructured it into disjoint field borrows, so a second borrow of the
    // whole would conflict with the vectors the same iteration is using.
    verdict: &mut ReadbackBuffer<f32>,
    verdict_cause: &mut ReadbackBuffer<i32>,
    rz_next: f32,
    rz_previous: f32,
) -> Result<PcgScalar, Fault> {
    let args = PcgBetaTermsArgs {
        // As `pcg_alpha`: the workspace's, because a handle names an allocation.
        value_out: verdict.span(0, 1),
        noise_out: verdict.span(1, 1),
        cause_out: verdict_cause.span(0, 1),
        rz_next,
        rz_previous,
        count: 1,
        seam_arena_count: 0,
    };
    // Safety: the three out-parameters outlive the dispatch.
    unsafe { device.launch("pcg.beta", &args, 1) }?;
    // THE MIRRORS, brought current before the verdict is read: taking a handle
    // above staled both by construction.
    verdict.download(device)?;
    verdict_cause.download(device)?;
    let value = verdict.host()[0];
    let noise = verdict.host()[1];
    let cause = verdict_cause.host()[0];
    let _ = noise;
    Ok(PcgScalar { value, cause })
}

/// Per-row inner-product terms, folded under `reduce`'s fixed shape.
///
/// TWO ARRAYS BECAUSE THE SHARED BODY YIELDS TWO NUMBERS. `pcg_dot_terms`
/// computes a row's product and the magnitude of that same product from one
/// expression and writes both through gathered outputs, so both have to be
/// bound. Only the product is folded here: the scale the curvature bound is
/// taken from is [`DotTerms::norm_product`], for the reason the module doc
/// gives. `reduce::sum` fixes the block shape, so the fold is independent of
/// how rayon split the work.
struct DotTerms<'a> {
    product: &'a mut ReadbackBuffer<f32>,
    magnitude: &'a mut ReadbackBuffer<f32>,
}

impl DotTerms<'_> {
    /// Refresh the PRODUCT mirror, which is the only one anything reads.
    ///
    /// THE MAGNITUDE ARRAY IS DELIBERATELY LEFT STALE. `pcg_dot_terms` is a
    /// shared body that yields both numbers from one expression, so the
    /// gathered output has to be bound and written; nothing on this side folds
    /// it, because the scale the curvature bound is taken from is
    /// [`norm_product`](Self::norm_product) rather than the per-row magnitude
    /// (see the module doc). Downloading it anyway would be `rows` floats a
    /// Newton iteration carried across the seam for no reader.
    ///
    /// Leaving it stale is loud rather than quiet: `ReadbackBuffer::host`
    /// panics on a mirror that a dispatch named and no download refreshed, so
    /// a future reader who folds it gets an assertion naming the buffer, not
    /// the contents from before the pass.
    fn download(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        self.product.download(device)
    }

    /// The per-row product array, as a handle, for the DEVICE fold that
    /// reduces it.
    ///
    /// It stales the mirror, as every `handle()` does, and that is correct
    /// here: nothing on this side reads the array any more. The fold's last
    /// level writes one float into a slot of the workspace's scalar buffer,
    /// and that slot is what the host reads.
    fn product_handle(&mut self) -> ppf_cts_compute::Handle {
        self.product.handle()
    }

    /// The magnitude array's handle, for the fused row that writes it.
    ///
    /// IT STILL HAS NO READER on this side, for the reason the note above
    /// gives; the fused body produces both numbers from one expression exactly
    /// as `pcg_dot_terms` does, so the output has to be bound either way.
    fn magnitude_handle(&mut self) -> ppf_cts_compute::Handle {
        self.magnitude.handle()
    }

    /// Rows, which is the extent every fill dispatches over and the length
    /// every fold of the product array reduces.
    fn rows(&self) -> usize {
        self.product.len()
    }

    /// Fill both arrays from the shared body.
    ///
    /// # Safety
    /// Both inputs must outlive the dispatch this appends.
    unsafe fn encode_fill(
        &mut self,
        encoder: &mut dyn Encoder,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        _n: usize,
    ) -> Result<(), Fault> {
        let rows = self.product.len() as u32;
        if rows == 0 {
            return Ok(());
        }
        let args = PcgDotTermsArgs {
            a,
            b,
            product: self.product.handle(),
            absolute_product: self.magnitude.handle(),
            count: rows,
            seam_arena_count: 0,
        };
        encoder.elements(&args, rows)
    }

    fn fill<D: Device>(
        &mut self,
        device: &mut D,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        _n: usize,
    ) -> Result<(), Fault> {
        let product = self.product.handle();
        let magnitude = self.magnitude.handle();
        let rows = self.product.len() as u32;
        if rows == 0 {
            return Ok(());
        }
        let args = PcgDotTermsArgs {
            a,
            b,
            product,
            absolute_product: magnitude,
            count: rows,
            seam_arena_count: 0,
        };
        // Safety: all four buffers are borrowed for the whole call.
        unsafe { device.launch("pcg.dot_terms", &args, rows) }?;
        Ok(())
    }

    /// `a . b` under the fixed fold shape.

    /// Same root projector, products and fold as the classic tail, encoded
    /// together after the original host observation of the factor fault word.
    unsafe fn ordered_root_dot<D: Device>(
        &mut self,
        device: &mut D,
        root: super::roots::View,
        residual: ppf_cts_compute::Handle,
        preconditioned: ppf_cts_compute::Handle,
        dim: usize,
        fold: &Buffer<f32>,
        scalars: &mut ReadbackBuffer<f32>,
    ) -> Result<f32, Fault> {
        let rows = self.rows();
        debug_assert!(rows != 0);
        let out = scalars.span(slot::FOLD_SCRATCH, 1);
        let source = self.product_handle();
        let _phase = super::phase::start("pcg.classic_tail.rz_packet");
        device.run("pcg.rigid.classic_tail.rz_packet", |encoder| unsafe {
            root.encode_project(encoder, preconditioned)?;
            self.encode_fill(encoder, residual, preconditioned, dim)?;
            encode_fold(encoder, source, rows, fold, out, false)
        })?;
        scalars.download(device)?;
        Ok(scalars.host()[slot::FOLD_SCRATCH])
    }

    /// `a . b`, REDUCED ON THE DEVICE.
    ///
    /// The fold lands in one scalar slot and only that slot is read back, so
    /// what crosses the seam is a result rather than the vector it was
    /// computed from.
    ///
    /// IT IS BIT-IDENTICAL TO THE EQUIVALENT HOST FOLD.
    /// `the_device_fold_agrees_with_the_host_fold_bit_for_bit` pins the two
    /// shapes together, so reducing here rather than on the host changes what is
    /// TRANSPORTED and not what is COMPUTED.
    fn dot<D: Device>(
        &mut self,
        device: &mut D,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        n: usize,
        fold: &Buffer<f32>,
        scalars: &mut ReadbackBuffer<f32>,
    ) -> Result<f32, Fault> {
        self.fill(device, a, b, n)?;
        let rows = self.product.len();
        if rows == 0 {
            return Ok(0.0);
        }
        let source = self.product.handle();
        let out = scalars.span(slot::FOLD_SCRATCH, 1);
        device.run("pcg.dot", |encoder| {
            // Safety: the product vector and the scratch outlive the call.
            unsafe { encode_fold(encoder, source, rows, fold, out, false) }
        })?;
        scalars.download(device)?;
        Ok(scalars.host()[slot::FOLD_SCRATCH])
    }

    /// `a . b` and the round-off scale the curvature bound is taken from,
    /// measured over the same two vectors.
    ///
    /// THE SCALE IS THE CAUCHY-SCHWARZ SURROGATE, not the fold beside the dot,
    /// for the reason [`norm_product`](Self::norm_product) records: this
    /// backend's per-row magnitude has already spent the row's own
    /// cancellation, so it is not the quantity the shared bound is written
    /// against.
    fn dot_with_bound<D: Device>(
        &mut self,
        device: &mut D,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        n: usize,
        fold: &Buffer<f32>,
        scalars: &mut ReadbackBuffer<f32>,
    ) -> Result<(f32, f32), Fault> {
        let dot = self.dot(device, a, b, n, fold, scalars)?;
        let scale = self.norm_product(device, a, b, n, fold, scalars)?;
        Ok((dot, scale))
    }

    /// `|a|_2 |b|_2`, the scale `pcg_curvature_noise` is given on a path whose
    /// operator apply produces no `sum|contribution|` beside its dot.
    ///
    /// IT IS A SANCTIONED SCALE, NOT AN APPROXIMATION OF THE FUSED ONE.
    /// `sqrt(p.p) * sqrt(Ap.Ap)` is formed from three plain inner products, and
    /// `pcg_curvature_noise` in `src/kernels/solver/pcg.kernel.cpp` names it in
    /// as many words: "A host-synchronizing loop whose SpMV produces no such sum
    /// passes the Cauchy-Schwarz surrogate |p|_2 |Ap|_2 >= sum|p_i (Ap)_i|
    /// instead, which errs toward truncating rather than aborting."
    ///
    /// The two square roots are taken in host double and narrowed only at the
    /// return. That is a host scalar and
    /// not device arithmetic, so the float64 ban does not reach it. Taking
    /// `sqrt(aa * bb)` in fp32 instead would be one multiplication rather than
    /// two roots and is not the same function: `aa * bb` can overflow fp32
    /// while neither factor nor the product of the roots does.
    fn norm_product<D: Device>(
        &mut self,
        device: &mut D,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        n: usize,
        fold: &Buffer<f32>,
        scalars: &mut ReadbackBuffer<f32>,
    ) -> Result<f32, Fault> {
        // Sequential on purpose: each fold is read back before the next is
        // encoded, so both can use the one scratch slot.
        let aa = self.dot(device, a, a, n, fold, scalars)?;
        let bb = self.dot(device, b, b, n, fold, scalars)?;
        Ok(((aa as f64).sqrt() * (bb as f64).sqrt()) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;
    use crate::driver::spmv;


    #[test]
    fn ordered_tail_root_dot_keeps_native_projection_and_fold_bits() {
        use crate::driver::roots::{ConstraintDomain, View};
        use crate::joint_root_motion_gpu::RootTargetGpu;

        let mut device = host_device();
        let mut targets = Buffer::none();
        targets.size(&mut device, 1, AllocLabel("ordered_tail.test.root")).unwrap();
        targets.write(&mut device, 0, &[RootTargetGpu {
            body: 0, enabled: 1, ..Default::default()
        }]).unwrap();

        for dim in [7usize, 257, 1025] {
            let root = View {
                targets: targets.handle(), count: 1, bodies: 1, base: (dim - 6) as u32,
                dim: dim as u32, constraint_domain: ConstraintDomain::default(),
            };
            let mut residual = Buffer::<f32>::none();
            let mut original = Buffer::<f32>::none();
            let mut packet = Buffer::<f32>::none();
            let mut fold = Buffer::<f32>::none();
            for buffer in [&mut residual, &mut original, &mut packet] {
                buffer.size(&mut device, dim, AllocLabel("ordered_tail.test.vector")).unwrap();
            }
            fold.size(&mut device, fold_scratch_len(dim), AllocLabel("ordered_tail.test.fold")).unwrap();
            let x: Vec<_> = (0..dim).map(|i| ((i % 17) as f32 - 5.0) / 11.0).collect();
            let z: Vec<_> = (0..dim).map(|i| ((i % 23) as f32 - 8.0) / 7.0).collect();
            residual.write(&mut device, 0, &x).unwrap();
            original.write(&mut device, 0, &z).unwrap();
            packet.write(&mut device, 0, &z).unwrap();

            let mut product = ReadbackBuffer::default();
            let mut magnitude = ReadbackBuffer::default();
            let mut scalar = ReadbackBuffer::default();
            product.size(&mut device, dim, AllocLabel("ordered_tail.test.product")).unwrap();
            magnitude.size(&mut device, dim, AllocLabel("ordered_tail.test.magnitude")).unwrap();
            scalar.size(&mut device, slot::COUNT, AllocLabel("ordered_tail.test.scalar")).unwrap();
            let mut terms = DotTerms { product: &mut product, magnitude: &mut magnitude };
            unsafe { root.project(&mut device, original.handle()).unwrap(); }
            let expected = terms.dot(&mut device, residual.handle(), original.handle(),
                dim, &fold, &mut scalar).unwrap();
            let actual = unsafe { terms.ordered_root_dot(&mut device, root, residual.handle(),
                packet.handle(), dim, &fold, &mut scalar) }.unwrap();
            assert_eq!(actual.to_bits(), expected.to_bits(), "size {dim}");

            let mut actual_z = vec![0.0f32; dim];
            let mut expected_z = vec![0.0f32; dim];
            packet.read(&mut device, 0, &mut actual_z).unwrap();
            original.read(&mut device, 0, &mut expected_z).unwrap();
            assert!(actual_z.iter().zip(expected_z).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(actual_z[dim - 6..].iter().all(|x| x.to_bits() == 0));
            for buffer in [&mut residual, &mut original, &mut packet, &mut fold] {
                buffer.free(&mut device).unwrap();
            }
            for buffer in [&mut product, &mut magnitude, &mut scalar] {
                buffer.free(&mut device).unwrap();
            }
        }
        targets.free(&mut device).unwrap();
    }

    #[test]
    fn rigid_iterate_export_copies_affine_rotation_not_old_output_or_zero() {
        use crate::driver::rigid_map::{RigidMap, Staged, RIGID_UNSET};
        use ppf_cts_compute::Handle;
        let mut device = host_device();
        let map = RigidMap { nrow: 2, n_bodies: 1, n_cloth: 1, dim: 9, body_base: 3,
            ..Default::default() };
        let mut staged = Staged::default();
        staged.vbody.size(&mut device, 2, AllocLabel("test.export.vbody")).unwrap();
        staged.vbody.at().copy_from_slice(&[RIGID_UNSET, 0]);
        staged.vbody.upload(&mut device).unwrap();
        staged.cloth_off.size(&mut device, 2, AllocLabel("test.export.cloth")).unwrap();
        staged.cloth_off.at().copy_from_slice(&[0, RIGID_UNSET]);
        staged.cloth_off.upload(&mut device).unwrap();
        staged.prot.size(&mut device, 6, AllocLabel("test.export.prot")).unwrap();
        staged.prot.write(&mut device, 0, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let mut reduction = crate::driver::pdrd::Reduction { map: &map, staged: &mut staged };
        let mut reduced = Buffer::<f32>::default();
        reduced.size(&mut device, 9, AllocLabel("test.export.reduced")).unwrap();
        let seed = [4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 0.125, -0.25, 0.5];
        reduced.write(&mut device, 0, &seed).unwrap();
        let mut x = Buffer::<f32>::default();
        x.size(&mut device, 9, AllocLabel("test.export.x")).unwrap();
        let mut rot = Buffer::<f32>::default();
        rot.size(&mut device, 3, AllocLabel("test.export.rot")).unwrap();
        rot.write(&mut device, 0, &[99.0; 3]).unwrap();

        unsafe {
            export_rigid_solve_iterate(&mut device, &mut reduction, reduced.handle(),
                x.handle(), Some(rot.handle()), RigidSolveInput::NewtonForce).unwrap();
        }
        let mut got = [0.0; 3];
        rot.read(&mut device, 0, &mut got).unwrap();
        assert_eq!(got, [0.125, -0.25, 0.5]);
        let mut full = [0.0; 6];
        x.read(&mut device, 0, &mut full).unwrap();
        assert_eq!(full, [4.0, 5.0, 6.0, 1.0, 2.5, 3.25]);

        // The same path also clears a previous nonzero rotation when the
        // solved seed's rotation is zero. Reduced-RHS output stays reduced.
        reduced.write(&mut device, 6, &[0.0; 3]).unwrap();
        unsafe {
            export_rigid_solve_iterate(&mut device, &mut reduction, reduced.handle(),
                x.handle(), Some(rot.handle()), RigidSolveInput::NewtonForce).unwrap();
        }
        rot.read(&mut device, 0, &mut got).unwrap();
        assert_eq!(got, [0.0; 3]);
        reduced.write(&mut device, 0, &seed).unwrap();
        unsafe {
            export_rigid_solve_iterate(&mut device, &mut reduction, reduced.handle(),
                x.handle(), None::<Handle>, RigidSolveInput::ReducedRhs).unwrap();
        }
        let mut inverse_column = [0.0; 9];
        x.read(&mut device, 0, &mut inverse_column).unwrap();
        assert_eq!(inverse_column, seed);
    }

    #[test]
    fn joint_batch8_first_stop_guards_state_and_frozen_fault() {
        let mut device = host_device();
        let mut slots = ReadbackBuffer::<f32>::default();
        slots.size(&mut device, 13, AllocLabel("test.joint_batch8.slots")).unwrap();
        // norm[0..2], initial[2..4], rz_next, rz_previous, beta,
        // relative, status, x, p, alpha.
        slots.seed(&mut device, &[0.01,0.01,1.0,2.0,1.0,1.0,17.0,0.0,0.0,9.0,7.0,5.0,11.0]).unwrap();
        let mut counter = ReadbackBuffer::<u32>::default();
        counter.size(&mut device, 3, AllocLabel("test.joint_batch8.counter")).unwrap();
        counter.seed(&mut device, &[0,0,123]).unwrap();
        let stop = super::super::kernels::JointPcgBatch8StopArgs {
            norm:slots.span(0,2), initial:slots.span(2,2), initial_all:3.0, groups:2,
            cloth_present:1, rz_next:slots.span(4,1), rz_previous:slots.span(5,1),
            beta:slots.span(6,1), relative:slots.span(7,1), status:slots.span(8,1),
            first_iteration:counter.span(0,1), current_iteration:counter.span(1,1),
            tolerance:0.1, max_iterations:8, extent:1, count:1, seam_arena_count:0,
        };
        unsafe { device.launch("test.batch8.stop", &stop, 1).unwrap(); }
        let add = super::super::kernels::FemPcgReducedBatchAddIndirectArgs {
            source:slots.span(10,1), destination:slots.span(9,1), coefficient:slots.span(11,1),
            sign:1.0, status:slots.span(8,1), extent:1, count:1, seam_arena_count:0,
        };
        let add_r = super::super::kernels::FemPcgReducedBatchAddIndirectArgs {
            source:slots.span(10,1), destination:slots.span(12,1), coefficient:slots.span(11,1),
            sign:-1.0, status:slots.span(8,1), extent:1, count:1, seam_arena_count:0,
        };
        let direction = super::super::kernels::FemPcgReducedDirectionArgs {
            z:slots.span(0,1), p:slots.span(10,1), beta:slots.span(6,1),
            status:slots.span(8,1), extent:1, count:1, seam_arena_count:0,
        };
        let ball = super::super::kernels::JointPcgBatch8BallArgs {
            count_rows:1, factors:slots.span(0,1), offsets:counter.span(0,1),
            published:counter.span(0,1), epoch:0, input:slots.span(0,1), output:slots.span(9,1),
            fault:counter.span(2,1), status:slots.span(8,1), count:1, seam_arena_count:0,
        };
        for _ in 0..7 {
            unsafe {
                device.launch("test.batch8.stop.tail", &stop, 1).unwrap();
                device.launch("test.batch8.add.tail", &add, 1).unwrap();
                device.launch("test.batch8.add_r.tail", &add_r, 1).unwrap();
                device.launch("test.batch8.direction.tail", &direction, 1).unwrap();
                device.launch("test.batch8.ball.tail", &ball, 1).unwrap();
            }
        }
        slots.download(&mut device).unwrap(); counter.download(&mut device).unwrap();
        assert_eq!(slots.host()[8],1.0); assert_eq!(slots.host()[9],9.0);
        assert_eq!(slots.host()[6],17.0); assert_eq!(slots.host()[10],7.0); assert_eq!(slots.host()[12],11.0); assert_eq!(counter.host(), &[1,1,123]);
    }

    #[test]
    fn joint_batch8_stopped_fault_guards_preserve_sentinels() {
        let mut device = host_device();
        let mut dummy = ReadbackBuffer::<f32>::default();
        dummy.size(&mut device, 1024, AllocLabel("test.guarded.dummy")).unwrap();
        dummy.seed(&mut device, &vec![0.0;1024]).unwrap();
        let mut status = ReadbackBuffer::<f32>::default();
        status.size(&mut device, 1, AllocLabel("test.guarded.status")).unwrap();
        status.seed(&mut device, &[1.0]).unwrap();
        let mut fault = ReadbackBuffer::<u32>::default();
        fault.size(&mut device, 1, AllocLabel("test.guarded.fault")).unwrap();
        fault.seed(&mut device, &[123]).unwrap();
        let args = super::super::kernels::ResidentGuardRestoringAngularFrozenApplyArgs {
            count_rows: 1,
            factors: dummy.handle(),
            offsets: dummy.handle(),
            published: dummy.handle(),
            epoch: 0,
            input: dummy.handle(),
            output: dummy.handle(),
            fault: fault.handle(),
            status: status.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        unsafe { device.launch("test.batch8.guard", &args, 1).unwrap(); }
        let args = super::super::kernels::ResidentGuardRestoringFemFrozenApplyArgs {
            record_count: 1,
            cached_hessian: dummy.handle(),
            cached_index: dummy.handle(),
            published_epoch: dummy.handle(),
            epoch: 0,
            input: dummy.handle(),
            output: dummy.handle(),
            fault: fault.handle(),
            status: status.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        unsafe { device.launch("test.batch8.guard", &args, 1).unwrap(); }
        let args = super::super::kernels::ResidentGuardJointAngularExtensionApplyArgs {
            records: dummy.handle(),
            twist_context: dummy.handle(),
            material_spin_context: dummy.handle(),
            limits: dummy.handle(),
            drives: dummy.handle(),
            drive_count: 0,
            drive_offsets: dummy.handle(),
            keys: dummy.handle(),
            state: dummy.handle(),
            reference: dummy.handle(),
            bodies: 0,
            base: 0,
            dim: 0,
            sample_time_s: 0.0,
            dt_s: 0.0,
            action: 0,
            input: dummy.handle(),
            tangent_cache: dummy.handle(),
            tangent_offsets: dummy.handle(),
            tangent_published: dummy.handle(),
            epoch: 0,
            output: dummy.handle(),
            fault: fault.handle(),
            status: status.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        unsafe { device.launch("test.batch8.guard", &args, 1).unwrap(); }
        let args = super::super::kernels::ResidentGuardJointFemExtensionApplyArgs {
            records: dummy.handle(),
            material_spin_context: dummy.handle(),
            record_count: 1,
            cache: dummy.handle(),
            vertex_to_free: dummy.handle(),
            vertex_count: 0,
            free_count: 0,
            rigid_base: 0,
            dim: 0,
            limits: dummy.handle(),
            drives: dummy.handle(),
            drive_count: 0,
            keys: dummy.handle(),
            sample_time_s: 0.0,
            dt: 0.0,
            action: 0,
            input: dummy.handle(),
            output: dummy.handle(),
            nodal_blocks: dummy.handle(),
            rigid_blocks: dummy.handle(),
            fault: fault.handle(),
            status: status.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        unsafe { device.launch("test.batch8.guard", &args, 1).unwrap(); }
        let args = super::super::kernels::ResidentGuardFttEvaluateArgs {
            tet: dummy.handle(),
            factor: dummy.handle(),
            weight: dummy.handle(),
            input: dummy.handle(),
            target: dummy.handle(),
            mask: dummy.handle(),
            response: dummy.handle(),
            fault: fault.handle(),
            lift: 0,
            status: status.handle(),
            extent: 1,
            count: 1,
            seam_arena_count: 0,
        };
        unsafe { device.launch("test.batch8.guard", &args, 1).unwrap(); }
        let args = super::super::kernels::ResidentGuardFttGatherArgs {
            adj_offset: dummy.handle(),
            adj: dummy.handle(),
            mask: dummy.handle(),
            tet: dummy.handle(),
            response: dummy.handle(),
            result: dummy.handle(),
            curvature: dummy.handle(),
            absolute: dummy.handle(),
            fault: fault.handle(),
            lift: 0,
            status: status.handle(),
            extent: 1,
            count: 1,
            seam_arena_count: 0,
        };
        unsafe { device.launch("test.batch8.guard", &args, 1).unwrap(); }
        fault.download(&mut device).unwrap();
        assert_eq!(fault.host()[0],123);
    }

    #[test]
    fn joint_batch8_tet_active_matches_original_and_faults() {
        fn f(device:&mut impl Device, values:&[f32])->ReadbackBuffer<f32> {
            let mut b=ReadbackBuffer::<f32>::default();
            b.size(device,values.len(),AllocLabel("test.tet.f32")).unwrap();
            b.seed(device,values).unwrap(); b
        }
        fn u(device:&mut impl Device, values:&[u32])->ReadbackBuffer<u32> {
            let mut b=ReadbackBuffer::<u32>::default();
            b.size(device,values.len(),AllocLabel("test.tet.u32")).unwrap();
            b.seed(device,values).unwrap(); b
        }
        let mut d=host_device();
        let mut tet=u(&mut d,&[0,1,2,3]); let mut mask=u(&mut d,&[0,0,0,0]);
        let mut offsets=u(&mut d,&[0,1,2,3,4]); let mut adj=u(&mut d,&[0,1,2,3]);
        let mut identity=vec![0.0;81]; for i in 0..9 {identity[9*i+i]=1.0;}
        let mut factor=f(&mut d,&identity); let mut weight=f(&mut d,&[1.0;9]);
        let mut input=f(&mut d,&[0.0,0.0,0.0, 1.0,2.0,3.0, 4.0,5.0,6.0, 7.0,8.0,9.0]);
        let mut status=f(&mut d,&[0.0]);
        let mut response_a=f(&mut d,&[0.0;11]); let mut response_b=f(&mut d,&[0.0;11]);
        let mut fault_a=u(&mut d,&[0;4]); let mut fault_b=u(&mut d,&[0;4]);
        let base=super::super::kernels::FttEvaluateArgs {
            tet:tet.handle(),factor:factor.handle(),weight:weight.handle(),input:input.handle(),
            target:input.handle(),mask:mask.handle(),response:response_a.handle(),fault:fault_a.handle(),
            lift:0,extent:1,count:1,seam_arena_count:0,
        };
        let guard=super::super::kernels::ResidentGuardFttEvaluateArgs {
            tet:tet.handle(),factor:factor.handle(),weight:weight.handle(),input:input.handle(),
            target:input.handle(),mask:mask.handle(),response:response_b.handle(),fault:fault_b.handle(),
            lift:0,status:status.handle(),extent:1,count:1,seam_arena_count:0,
        };
        unsafe {d.launch("test.tet.eval",&base,1).unwrap();d.launch("test.tet.eval.guard",&guard,1).unwrap();}
        response_a.download(&mut d).unwrap();response_b.download(&mut d).unwrap();
        assert_eq!(response_a.host(),response_b.host());
        assert!(response_a.host().iter().any(|x|*x!=0.0));
        let mut result_a=f(&mut d,&[0.0;12]);let mut result_b=f(&mut d,&[0.0;12]);
        let mut curve_a=f(&mut d,&[0.0;4]);let mut curve_b=f(&mut d,&[0.0;4]);
        let mut abs_a=f(&mut d,&[0.0;4]);let mut abs_b=f(&mut d,&[0.0;4]);
        let ga=super::super::kernels::FttGatherArgs {
            adj_offset:offsets.handle(),adj:adj.handle(),mask:mask.handle(),tet:tet.handle(),
            response:response_a.handle(),result:result_a.handle(),curvature:curve_a.handle(),
            absolute:abs_a.handle(),fault:fault_a.handle(),lift:0,extent:4,count:4,seam_arena_count:0,
        };
        let gb=super::super::kernels::ResidentGuardFttGatherArgs {
            adj_offset:offsets.handle(),adj:adj.handle(),mask:mask.handle(),tet:tet.handle(),
            response:response_b.handle(),result:result_b.handle(),curvature:curve_b.handle(),
            absolute:abs_b.handle(),fault:fault_b.handle(),lift:0,status:status.handle(),
            extent:4,count:4,seam_arena_count:0,
        };
        unsafe {d.launch("test.tet.gather",&ga,4).unwrap();d.launch("test.tet.gather.guard",&gb,4).unwrap();}
        result_a.download(&mut d).unwrap();result_b.download(&mut d).unwrap();
        curve_a.download(&mut d).unwrap();curve_b.download(&mut d).unwrap();
        abs_a.download(&mut d).unwrap();abs_b.download(&mut d).unwrap();
        assert_eq!(result_a.host(),result_b.host());assert_eq!(curve_a.host(),curve_b.host());
        assert_eq!(abs_a.host(),abs_b.host());
        identity[0]=f32::NAN;factor.seed(&mut d,&identity).unwrap();
        unsafe {d.launch("test.tet.bad",&base,1).unwrap();d.launch("test.tet.bad.guard",&guard,1).unwrap();}
        fault_a.download(&mut d).unwrap();fault_b.download(&mut d).unwrap();
        assert_eq!(fault_a.host(),fault_b.host());assert_eq!(fault_a.host()[0],5);
        fault_a.seed(&mut d,&[0;4]).unwrap();fault_b.seed(&mut d,&[0;4]).unwrap();
        let mut bad=[0.0;11];bad[0]=f32::NAN;response_a.seed(&mut d,&bad).unwrap();response_b.seed(&mut d,&bad).unwrap();
        unsafe {d.launch("test.tet.gather.bad",&ga,4).unwrap();d.launch("test.tet.gather.bad.guard",&gb,4).unwrap();}
        fault_a.download(&mut d).unwrap();fault_b.download(&mut d).unwrap();
        assert_eq!(fault_a.host(),fault_b.host());assert!(fault_a.host().iter().any(|x|*x==6));
    }

    #[test]
    fn joint_batch8_latches_first_stop_across_two_replays() {
        let mut d=host_device();
        let mut norm=ReadbackBuffer::<f32>::default();
        norm.size(&mut d,1,AllocLabel("test.batch8.norm")).unwrap();norm.seed(&mut d,&[1.0]).unwrap();
        let mut slots=ReadbackBuffer::<f32>::default();
        slots.size(&mut d,6,AllocLabel("test.batch8.scalar")).unwrap();
        slots.seed(&mut d,&[1.0,1.0,1.0,0.0,0.0,0.0]).unwrap();
        let mut counters=ReadbackBuffer::<u32>::default();
        counters.size(&mut d,2,AllocLabel("test.batch8.counters")).unwrap();counters.seed(&mut d,&[0,0]).unwrap();
        let args=super::super::kernels::JointPcgBatch8StopArgs {
            norm:norm.handle(),initial:slots.span(0,1),initial_all:1.0,groups:1,cloth_present:1,
            rz_next:slots.span(1,1),rz_previous:slots.span(2,1),beta:slots.span(3,1),
            relative:slots.span(4,1),status:slots.span(5,1),first_iteration:counters.span(0,1),
            current_iteration:counters.span(1,1),tolerance:0.1,max_iterations:16,
            extent:1,count:1,seam_arena_count:0,
        };
        let region=d.record("test.batch8.two_replays",|encoder| {
            for _ in 0..8 {unsafe {encoder.elements(&args,1)?;}}
            Ok(())
        }).unwrap();
        d.replay(&region,1).unwrap();counters.download(&mut d).unwrap();slots.download(&mut d).unwrap();
        assert_eq!(counters.host(),&[0,8]);assert_eq!(slots.host()[5],0.0);
        norm.seed(&mut d,&[0.01]).unwrap();
        d.replay(&region,1).unwrap();d.release(region);
        counters.download(&mut d).unwrap();slots.download(&mut d).unwrap();
        assert_eq!(counters.host(),&[9,9]);assert_eq!(slots.host()[5],1.0);
    }

    #[test]
    fn joint_batch8_frozen_fem_active_matches_original() {
        fn f(d:&mut impl Device, v:&[f32])->ReadbackBuffer<f32> {
            let mut b=ReadbackBuffer::default(); b.size(d,v.len(),AllocLabel("test.frozen.f32")).unwrap(); b.seed(d,v).unwrap(); b
        }
        fn u(d:&mut impl Device, v:&[u32])->ReadbackBuffer<u32> {
            let mut b=ReadbackBuffer::default(); b.size(d,v.len(),AllocLabel("test.frozen.u32")).unwrap(); b.seed(d,v).unwrap(); b
        }
        let mut d=host_device();
        let mut h=vec![0.0f32;576];
        for i in 0..24 { h[24*i+i]=2.0; }
        let mut h=f(&mut d,&h);
        let mut idx=u(&mut d,&(0u32..24).collect::<Vec<_>>());
        let mut epoch=u(&mut d,&[3]);
        let mut input=f(&mut d,&(1..=24).map(|x|x as f32).collect::<Vec<_>>());
        let mut status=f(&mut d,&[0.0]);
        let mut a=f(&mut d,&[0.0;24]); let mut b=f(&mut d,&[0.0;24]);
        let mut fa=u(&mut d,&[0]); let mut fb=u(&mut d,&[0]);
        for expected_epoch in [3,4] {
            let original=super::super::kernels::RestoringFemFrozenApplyArgs {
                record_count:1,cached_hessian:h.handle(),cached_index:idx.handle(),published_epoch:epoch.handle(),
                epoch:expected_epoch,input:input.handle(),output:a.handle(),fault:fa.handle(),count:1,seam_arena_count:0,
            };
            let guarded=super::super::kernels::ResidentGuardRestoringFemFrozenApplyArgs {
                record_count:1,cached_hessian:h.handle(),cached_index:idx.handle(),published_epoch:epoch.handle(),
                epoch:expected_epoch,input:input.handle(),output:b.handle(),fault:fb.handle(),status:status.handle(),count:1,seam_arena_count:0,
            };
            unsafe {d.launch("test.frozen.original",&original,1).unwrap(); d.launch("test.frozen.guarded",&guarded,1).unwrap();}
            a.download(&mut d).unwrap(); b.download(&mut d).unwrap(); fa.download(&mut d).unwrap(); fb.download(&mut d).unwrap();
            assert_eq!(a.host(),b.host()); assert_eq!(fa.host(),fb.host());
            if expected_epoch == 3 {assert_eq!(a.host()[0],2.0); assert_eq!(a.host()[23],48.0); assert_eq!(fa.host()[0],0);}
            else {assert_ne!(fa.host()[0],0);}
        }
    }

    #[test]
    fn joint_batch8_frozen_angular_active_matches_original() {
        fn f(d:&mut impl Device, v:&[f32])->ReadbackBuffer<f32> {
            let mut b=ReadbackBuffer::default(); b.size(d,v.len(),AllocLabel("test.angular.f32")).unwrap(); b.seed(d,v).unwrap(); b
        }
        fn u(d:&mut impl Device, v:&[u32])->ReadbackBuffer<u32> {
            let mut b=ReadbackBuffer::default(); b.size(d,v.len(),AllocLabel("test.angular.u32")).unwrap(); b.seed(d,v).unwrap(); b
        }
        let mut d=host_device();
        let mut factors=vec![0.0f32;21];
        for i in 0..3 { factors[6*i+i]=1.0; factors[6*i+i+3]=-1.0; factors[18+i]=2.0; }
        let mut factors=f(&mut d,&factors); let mut offsets=u(&mut d,&[0,6]); let mut published=u(&mut d,&[3]);
        let mut input=f(&mut d,&(1..=12).map(|x|x as f32).collect::<Vec<_>>());
        let mut status=f(&mut d,&[0.0]); let mut a=f(&mut d,&[0.0;12]); let mut b=f(&mut d,&[0.0;12]);
        let mut fa=u(&mut d,&[0]); let mut fb=u(&mut d,&[0]);
        for epoch in [3,4] {
            let original=super::super::kernels::RestoringAngularFrozenApplyArgs {
                count_rows:1,factors:factors.handle(),offsets:offsets.handle(),published:published.handle(),epoch,
                input:input.handle(),output:a.handle(),fault:fa.handle(),count:1,seam_arena_count:0,
            };
            let guarded=super::super::kernels::ResidentGuardRestoringAngularFrozenApplyArgs {
                count_rows:1,factors:factors.handle(),offsets:offsets.handle(),published:published.handle(),epoch,
                input:input.handle(),output:b.handle(),fault:fb.handle(),status:status.handle(),count:1,seam_arena_count:0,
            };
            unsafe {d.launch("test.angular.original",&original,1).unwrap(); d.launch("test.angular.guarded",&guarded,1).unwrap();}
            a.download(&mut d).unwrap(); b.download(&mut d).unwrap(); fa.download(&mut d).unwrap(); fb.download(&mut d).unwrap();
            assert_eq!(a.host(),b.host()); assert_eq!(fa.host(),fb.host());
            if epoch == 3 {assert_ne!(a.host()[3],0.0); assert_eq!(fa.host()[0],0);}
            else {assert_ne!(fa.host()[0],0);}
        }
    }

    #[test]
    fn joint_batch8_rigid_extension_active_matches_original() {
        use ppf_cts_compute::Buffer;
        use crate::{continuous_twist::AngularTwistEvalContext, data::PdrdRigidState,
            joint_angular_control_gpu::AngularControlGpu,
            joint_angular_extension_gpu::{AngularSoftLimitGpuV1, AngularDriveGpuV1, AngularDriveDeviceKeyV1},
            material_spin_history::MaterialSpinContext};
        fn upload<T:ppf_cts_compute::Pod>(d:&mut impl Device,v:&[T])->Buffer<T> {
            let mut b=Buffer::<T>::default(); b.size(d,v.len(),AllocLabel("test.ext.input")).unwrap(); b.write(d,0,v).unwrap(); b
        }
        let mut d=host_device();
        let row=AngularControlGpu {body_a:0,body_b:1,enabled:1,coordinate_model:1,
            frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],rest_relative_wxyz:[1.,0.,0.,0.],
            stiffness_nm_per_rad:[0.;3],damping_nms_per_rad:[0.;3]};
        let mut records=upload(&mut d,&[row]);
        let context=upload(&mut d,&[AngularTwistEvalContext::default()]);
        let material=upload(&mut d,&[MaterialSpinContext::default()]);
        let limits=upload(&mut d,&[AngularSoftLimitGpuV1 {joint_index:0,enabled_axis_mask:1,
            lower_rad:[-0.2,-1.,-1.],upper_rad:[-0.1,1.,1.],stiffness_nm_per_rad:[20.,0.,0.],
            swing_cone_enabled:0,swing_cone_half_angle_rad:0.,swing_cone_stiffness_nm_per_rad:0.}]);
        let mut drives=Buffer::<AngularDriveGpuV1>::default(); drives.size(&mut d,1,AllocLabel("test.ext.drive")).unwrap();
        let mut keys=Buffer::<AngularDriveDeviceKeyV1>::default(); keys.size(&mut d,1,AllocLabel("test.ext.key")).unwrap();
        let drive_offsets=upload(&mut d,&[0u32,0u32]);
        let identity=[1.,0.,0.,0.,1.,0.,0.,0.,1.];
        let states=upload(&mut d,&[PdrdRigidState{rotation:identity,..Default::default()};2]);
        let input=upload(&mut d,&[0f32,0.,0.,1.,0.,0.,0.,0.,0.,-1.,0.,0.]);
        let status=upload(&mut d,&[0f32]);
        let mut a=upload(&mut d,&[0f32;12]); let mut b=upload(&mut d,&[0f32;12]);
        let mut fa=upload(&mut d,&[0u32]); let mut fb=upload(&mut d,&[0u32]);
        let mut tangent=upload(&mut d,&[0f32;22]);
        let mut tangent_offsets=upload(&mut d,&[u32::MAX;2]);
        let mut tangent_epoch=upload(&mut d,&[0u32]);
        for invalid in [false,true] {
            a.write(&mut d,0,&[0f32;12]).unwrap(); b.write(&mut d,0,&[0f32;12]).unwrap();
            fa.write(&mut d,0,&[0u32]).unwrap(); fb.write(&mut d,0,&[0u32]).unwrap();
            if invalid { records.write(&mut d,0,&[AngularControlGpu { frame_a_wxyz:[0.;4],..row }]).unwrap(); }
            let orig=super::super::kernels::JointAngularExtensionApplyArgs {
                records:records.handle(),twist_context:context.handle(),material_spin_context:material.handle(),limits:limits.handle(),
                drives:drives.handle(),drive_count:0,drive_offsets:drive_offsets.handle(),keys:keys.handle(),state:states.handle(),reference:states.handle(),
                bodies:2,base:0,dim:12,sample_time_s:0.01,dt_s:0.01,action:1,input:input.handle(),tangent_cache:input.handle(),tangent_offsets:keys.handle(),tangent_published:keys.handle(),epoch:0,output:a.handle(),
                fault:fa.handle(),count:1,seam_arena_count:0,
            };
            let guard=super::super::kernels::ResidentGuardJointAngularExtensionApplyArgs {
                records:records.handle(),twist_context:context.handle(),material_spin_context:material.handle(),limits:limits.handle(),
                drives:drives.handle(),drive_count:0,drive_offsets:drive_offsets.handle(),keys:keys.handle(),state:states.handle(),reference:states.handle(),
                bodies:2,base:0,dim:12,sample_time_s:0.01,dt_s:0.01,action:1,input:input.handle(),tangent_cache:input.handle(),tangent_offsets:keys.handle(),tangent_published:keys.handle(),epoch:0,output:b.handle(),
                fault:fb.handle(),status:status.handle(),count:1,seam_arena_count:0,
            };
            unsafe { d.launch("test.ext.original",&orig,1).unwrap(); d.launch("test.ext.guarded",&guard,1).unwrap(); }
            let mut oa=[0f32;12]; let mut ob=[0f32;12]; let mut ca=[0u32]; let mut cb=[0u32];
            a.read(&mut d,0,&mut oa).unwrap(); b.read(&mut d,0,&mut ob).unwrap();
            fa.read(&mut d,0,&mut ca).unwrap(); fb.read(&mut d,0,&mut cb).unwrap();
            assert_eq!(oa,ob); assert_eq!(ca,cb);
            if invalid { assert_ne!(ca[0],0); } else {
                assert_eq!(ca[0],0); assert!(oa.iter().any(|x| *x != 0.0));
                // The frozen extension tangent must match the uncached HVP,
                // and an older epoch must fall back to the uncached operator.
                tangent_epoch.write(&mut d,0,&[0u32]).unwrap();
                let build=super::super::kernels::JointAngularExtensionApplyArgs {
                    records:records.handle(),twist_context:context.handle(),material_spin_context:material.handle(),limits:limits.handle(),
                    drives:drives.handle(),drive_count:0,drive_offsets:drive_offsets.handle(),keys:keys.handle(),state:states.handle(),reference:states.handle(),
                    bodies:2,base:0,dim:12,sample_time_s:0.01,dt_s:0.01,action:6,input:input.handle(),tangent_cache:tangent.handle(),
                    tangent_offsets:tangent_offsets.handle(),tangent_published:tangent_epoch.handle(),epoch:7,output:a.handle(),fault:fa.handle(),count:1,seam_arena_count:0,
                };
                unsafe { d.launch("test.ext.original",&build,1).unwrap(); }
                let mut published=[0u32];tangent_epoch.read(&mut d,0,&mut published).unwrap();assert_eq!(published,[7]);
                a.write(&mut d,0,&[0f32;12]).unwrap(); fa.write(&mut d,0,&[0u32]).unwrap();
                let cached=super::super::kernels::JointAngularExtensionApplyArgs {
                    records:records.handle(),twist_context:context.handle(),material_spin_context:material.handle(),limits:limits.handle(),
                    drives:drives.handle(),drive_count:0,drive_offsets:drive_offsets.handle(),keys:keys.handle(),state:states.handle(),reference:states.handle(),
                    bodies:2,base:0,dim:12,sample_time_s:0.01,dt_s:0.01,action:7,input:input.handle(),tangent_cache:tangent.handle(),
                    tangent_offsets:tangent_offsets.handle(),tangent_published:tangent_epoch.handle(),epoch:7,output:a.handle(),fault:fa.handle(),count:1,seam_arena_count:0,
                };
                unsafe { d.launch("test.ext.original",&cached,1).unwrap(); }
                let mut cached_out=[0f32;12];a.read(&mut d,0,&mut cached_out).unwrap();
                for i in 0..12 {assert!((cached_out[i]-oa[i]).abs()<=2e-5,"extension tangent mismatch at {i}");}
                tangent_epoch.write(&mut d,0,&[8u32]).unwrap();
                a.write(&mut d,0,&[0f32;12]).unwrap(); fa.write(&mut d,0,&[0u32]).unwrap();
                unsafe { d.launch("test.ext.original",&cached,1).unwrap(); }
                let mut fallback=[0f32;12];a.read(&mut d,0,&mut fallback).unwrap();
                assert_eq!(fallback,oa,"stale extension tangent epoch must fall back to the live HVP");
            }
        }
    }

    #[test]
    fn joint_batch8_fem_extension_active_matches_original() { unsafe {
        use ppf_cts_compute::Buffer;
        use crate::{soft_joint_gpu::{FemJointGpu,FemFrameGpu,FEM_TET,FEM_WORLD},
            joint_angular_control_gpu::AngularControlGpu,
            joint_angular_extension_gpu::{AngularSoftLimitGpuV1,AngularDriveGpuV1,AngularDriveDeviceKeyV1},
            material_spin_history::MaterialSpinContext};
        fn put<T:ppf_cts_compute::Pod>(d:&mut impl Device,v:&[T])->Buffer<T> {
            let mut b=Buffer::default(); b.size(d,v.len(),AllocLabel("test.femext")).unwrap(); b.write(d,0,v).unwrap(); b
        }
        let mut d=host_device();
        let identity=[1.,0.,0.,0.,1.,0.,0.,0.,1.];
        let mut row:FemJointGpu=std::mem::zeroed(); row.enabled=1; row.a.kind=FEM_WORLD;
        row.b.kind=FEM_TET; row.b.vertices=[0,1,2,3]; row.b.dm_inv=identity;
        row.b.anchor_bary=[1.,0.,0.,0.]; row.b.center_bary=[1.,0.,0.,0.]; row.b.rest_rotation=identity;
        row.control=AngularControlGpu{body_a:0,body_b:1,enabled:1,coordinate_model:1,
            frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],rest_relative_wxyz:[1.,0.,0.,0.],
            stiffness_nm_per_rad:[0.;3],damping_nms_per_rad:[0.;3]};
        let records=put(&mut d,&[row]);
        let xyz=put(&mut d,&[0f32,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.]);
        let dummy=put(&mut d,&[0f32;36]);
        let frames:[FemFrameGpu;27]=std::mem::zeroed(); let mut cache=put(&mut d,&frames);
        let mut prepare_fault=put(&mut d,&[0u32]);
        let prepare=super::super::kernels::JointFemPrepareArgs { records:records.handle(),record_count:1,
            current_xyz:xyz.handle(),reference_xyz:xyz.handle(),vertex_count:4,rigid_current:dummy.handle(),
            rigid_reference:dummy.handle(),bodies:0,cache:cache.handle(),fault:prepare_fault.handle(),count:1,seam_arena_count:0 };
        d.launch("test.femext.prepare",&prepare,1).unwrap();
        let mut code=[0u32]; prepare_fault.read(&mut d,0,&mut code).unwrap(); assert_eq!(code,[0]);
        let mut frames:[FemFrameGpu;4]=std::mem::zeroed(); cache.read(&mut d,0,&mut frames).unwrap();
        assert_eq!(frames.map(|x|x.status),[0,0,0,0]);
        let material=put(&mut d,&[MaterialSpinContext::default()]);
        let mapping=put(&mut d,&[0u32,1,2,3]);
        let limits=put(&mut d,&[AngularSoftLimitGpuV1 {joint_index:0,enabled_axis_mask:1,
            lower_rad:[-0.2,-1.,-1.],upper_rad:[-0.1,1.,1.],stiffness_nm_per_rad:[20.,0.,0.],
            swing_cone_enabled:0,swing_cone_half_angle_rad:0.,swing_cone_stiffness_nm_per_rad:0.}]);
        let mut drives=Buffer::<AngularDriveGpuV1>::default(); drives.size(&mut d,1,AllocLabel("test.femext.drive")).unwrap();
        let mut keys=Buffer::<AngularDriveDeviceKeyV1>::default(); keys.size(&mut d,1,AllocLabel("test.femext.key")).unwrap();
        let input=put(&mut d,&[0f32,0.,0.,0.,0.,0.,0.,0.,1.,0.,-1.,0.]);
        let mut status=put(&mut d,&[0f32]); let mut a=put(&mut d,&[0f32;12]); let mut b=put(&mut d,&[0f32;12]);
        let mut fa=put(&mut d,&[0u32]); let mut fb=put(&mut d,&[0u32]);
        let mut c=put(&mut d,&[0f32;12]); let mut fc=put(&mut d,&[0u32]);
        for invalid in [false,true] {
            a.write(&mut d,0,&[0f32;12]).unwrap(); b.write(&mut d,0,&[0f32;12]).unwrap();
            fa.write(&mut d,0,&[0u32]).unwrap(); fb.write(&mut d,0,&[0u32]).unwrap();
            if invalid {frames[1].status=259;cache.write(&mut d,0,&frames).unwrap();}
            let orig=super::super::kernels::JointFemExtensionApplyArgs {
                records:records.handle(),material_spin_context:material.handle(),record_count:1,cache:cache.handle(),
                vertex_to_free:mapping.handle(),vertex_count:4,free_count:4,rigid_base:12,dim:12,limits:limits.handle(),
                drives:drives.handle(),drive_count:0,keys:keys.handle(),sample_time_s:0.01,dt:0.01,action:1,
                input:input.handle(),output:a.handle(),nodal_blocks:dummy.handle(),rigid_blocks:dummy.handle(),fault:fa.handle(),count:1,seam_arena_count:0,
            };
            let guard=super::super::kernels::ResidentGuardJointFemExtensionApplyArgs {
                records:records.handle(),material_spin_context:material.handle(),record_count:1,cache:cache.handle(),
                vertex_to_free:mapping.handle(),vertex_count:4,free_count:4,rigid_base:12,dim:12,limits:limits.handle(),
                drives:drives.handle(),drive_count:0,keys:keys.handle(),sample_time_s:0.01,dt:0.01,action:1,
                input:input.handle(),output:b.handle(),nodal_blocks:dummy.handle(),rigid_blocks:dummy.handle(),fault:fb.handle(),status:status.handle(),count:1,seam_arena_count:0,
            };
            d.launch("test.femext.original",&orig,1).unwrap(); d.launch("test.femext.guarded",&guard,1).unwrap();
            let mut oa=[0f32;12];let mut ob=[0f32;12];let mut ca=[0u32];let mut cb=[0u32];
            a.read(&mut d,0,&mut oa).unwrap();b.read(&mut d,0,&mut ob).unwrap();fa.read(&mut d,0,&mut ca).unwrap();fb.read(&mut d,0,&mut cb).unwrap();
            assert_eq!(oa,ob);assert_eq!(ca,cb);
            if invalid {assert_eq!(ca[0],259);} else {assert_eq!(ca[0],0);assert!(oa.iter().any(|x|*x!=0.0));}
        }
        frames[1].status=0; cache.write(&mut d,0,&frames).unwrap();
        fa.write(&mut d,0,&[0u32]).unwrap(); fb.write(&mut d,0,&[0u32]).unwrap();
        let mut orig=super::super::kernels::JointFemExtensionApplyArgs {
            records:records.handle(),material_spin_context:material.handle(),record_count:1,cache:cache.handle(),
            vertex_to_free:mapping.handle(),vertex_count:4,free_count:4,rigid_base:12,dim:12,limits:limits.handle(),
            drives:drives.handle(),drive_count:0,keys:keys.handle(),sample_time_s:0.01,dt:0.01,action:(3<<3)|6,
            input:input.handle(),output:a.handle(),nodal_blocks:dummy.handle(),rigid_blocks:dummy.handle(),fault:fa.handle(),count:1,seam_arena_count:0,
        };
        d.launch("test.femext.cache_build",&orig,1).unwrap(); fa.read(&mut d,0,&mut code).unwrap(); assert_eq!(code,[0]);
        for epoch in [3,4] {
            a.write(&mut d,0,&[0f32;12]).unwrap(); b.write(&mut d,0,&[0f32;12]).unwrap();
            fa.write(&mut d,0,&[0u32]).unwrap(); fb.write(&mut d,0,&[0u32]).unwrap();
            c.write(&mut d,0,&[0f32;12]).unwrap(); fc.write(&mut d,0,&[0u32]).unwrap();
            orig.action=(epoch<<3)|7;
            let guard=super::super::kernels::ResidentGuardJointFemExtensionApplyArgs {
                records:records.handle(),material_spin_context:material.handle(),record_count:1,cache:cache.handle(),
                vertex_to_free:mapping.handle(),vertex_count:4,free_count:4,rigid_base:12,dim:12,limits:limits.handle(),
                drives:drives.handle(),drive_count:0,keys:keys.handle(),sample_time_s:0.01,dt:0.01,action:(epoch<<3)|7,
                input:input.handle(),output:b.handle(),nodal_blocks:dummy.handle(),rigid_blocks:dummy.handle(),fault:fb.handle(),status:status.handle(),count:1,seam_arena_count:0,
            };
            d.launch("test.femext.cached_original",&orig,1).unwrap(); d.launch("test.femext.cached_guarded",&guard,1).unwrap();
            let rows=super::super::kernels::JointFemExtensionFrozenRowArgs {
                records:records.handle(),record_count:1,cache:cache.handle(),vertex_to_free:mapping.handle(),
                vertex_count:4,free_count:4,rigid_base:12,dim:12,limits:limits.handle(),
                drives:drives.handle(),drive_count:0,dt:0.01,token:epoch,input:input.handle(),output:c.handle(),
                fault:fc.handle(),stopped:status.handle(),use_stopped:1,extent:24,count:24,seam_arena_count:0,
            };
            d.launch("test.femext.cached_rows",&rows,24).unwrap();
            let mut oa=[0f32;12]; let mut ob=[0f32;12]; let mut ca=[0u32]; let mut cb=[0u32];
            a.read(&mut d,0,&mut oa).unwrap(); b.read(&mut d,0,&mut ob).unwrap(); fa.read(&mut d,0,&mut ca).unwrap(); fb.read(&mut d,0,&mut cb).unwrap();
            let mut oc=[0f32;12]; let mut cc=[0u32]; c.read(&mut d,0,&mut oc).unwrap(); fc.read(&mut d,0,&mut cc).unwrap();
            for (x,y) in oa.iter().zip(oc) {assert!((x-y).abs()<1e-5);} assert_eq!(ca,cc);
            status.write(&mut d,0,&[1f32]).unwrap();fc.write(&mut d,0,&[99u32]).unwrap();
            d.launch("test.femext.stopped_rows",&rows,24).unwrap();
            let mut after=[0f32;12];let mut sentinel=[0u32];c.read(&mut d,0,&mut after).unwrap();fc.read(&mut d,0,&mut sentinel).unwrap();
            assert_eq!(after,oc);assert_eq!(sentinel,[99]);status.write(&mut d,0,&[0f32]).unwrap();
            assert_eq!(oa,ob); assert_eq!(ca,cb);
            if epoch == 3 {assert_eq!(ca[0],0);assert!(oa.iter().any(|x|*x!=0.0));} else {assert_ne!(ca[0],0);}
        }
    }}

    #[test]
    fn joint_batch8_prefix_transition_preserves_rz_and_iteration() {
        let mut d = host_device();
        let mut norm = ReadbackBuffer::<f32>::default();
        norm.size(&mut d, 1, AllocLabel("test.prefix.norm")).unwrap();
        norm.seed(&mut d, &[1.0]).unwrap();
        let mut slots = ReadbackBuffer::<f32>::default();
        slots.size(&mut d, 8, AllocLabel("test.prefix.scalar")).unwrap();
        // initial, rz_next, rz_previous, beta, relative, status, first, current
        slots.seed(&mut d, &[1.0, 7.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let prefix = super::super::kernels::JointPcgGroupStopBetaArgs {
            norm:norm.handle(), initial:slots.span(0,1), initial_all:1.0,
            groups:1, cloth_present:1, rz_next:slots.span(1,1), rz_previous:slots.span(2,1),
            beta:slots.span(3,1), relative:slots.span(4,1), status:slots.span(5,1),
            tolerance:0.1, iteration:8, max_iterations:16, extent:1, count:1, seam_arena_count:0,
        };
        unsafe { d.launch("test.prefix.stop", &prefix, 1).unwrap(); }
        slots.download(&mut d).unwrap();
        assert_eq!(slots.host()[2], 7.0);
        assert_eq!(slots.host()[5], 0.0);
        let mut transition = slots.host().to_vec();
        transition[6] = f32::from_bits(0);
        transition[7] = f32::from_bits(8);
        slots.seed(&mut d, &transition).unwrap();
        norm.seed(&mut d, &[0.01]).unwrap();
        let batch = super::super::kernels::JointPcgBatch8StopArgs {
            norm:norm.handle(), initial:slots.span(0,1), initial_all:1.0,
            groups:1, cloth_present:1, rz_next:slots.span(1,1), rz_previous:slots.span(2,1),
            beta:slots.span(3,1), relative:slots.span(4,1), status:slots.span(5,1),
            first_iteration:slots.span(6,1), current_iteration:slots.span(7,1),
            tolerance:0.1, max_iterations:16, extent:1, count:1, seam_arena_count:0,
        };
        unsafe { d.launch("test.prefix.first_batch_stop", &batch, 1).unwrap(); }
        slots.download(&mut d).unwrap();
        assert_eq!(slots.host()[2], 7.0);
        assert_eq!(slots.host()[5], 1.0);
        assert_eq!(slots.host()[6].to_bits(), 9);
        assert_eq!(slots.host()[7].to_bits(), 9);
    }

    #[test]
    #[ignore = "release dispatch-overhead timing, run alone with --nocapture"]
    fn benchmark_pcg_dispatch_overhead() {
        use std::hash::{Hash, Hasher};

        // The production driver calls from outside Rayon. Installing this test
        // in a pool would omit the queue-entry overhead it measures.
        let threads = rayon::current_num_threads();
        assert!(rayon::current_thread_index().is_none());
        for rows in [1025, 8192, 81920, 600001] {
            let matrix = spd(rows);
            let n = 3 * rows as usize;
            let mut device = host_device();
            let resident = matrix.upload(&mut device);
            let diagonal = staged_block(&mut device, &matrix.zero_diagonal, "test.zero");
            let inverse = matrix.inverse(&mut device);
            let b = staged_block(
                &mut device,
                &(0..n).map(|i| (i % 17) as f32 / 17.0).collect::<Vec<_>>(),
                "test.b",
            );
            let zero = vec![0.0; n];
            let mut x = staged_block(&mut device, &zero, "test.x");
            let op = matrix.op(&resident, diagonal.handle());
            let mut work = matrix.work(&mut device);
            let mut expected = None;
            let mut samples = Vec::new();
            for trial in 0..9 {
                let mut elapsed = 0.0;
                for _ in 0..8 {
                    x.write(&mut device, 0, &zero).unwrap();
                    let start = std::time::Instant::now();
                    let report = solve(
                        &mut device, &op, inverse.handle(), b.handle(),
                        x.handle(), &mut work, None, 1.0e-5, 100,
                    ).unwrap();
                    elapsed += start.elapsed().as_secs_f64() * 1e3;
                    assert_eq!(report.outcome, Outcome::Converged);
                    let value: Vec<u32> = read_back(&mut device, &x, n)
                        .into_iter().map(f32::to_bits).collect();
                    let answer = (value, report.iterations, report.relative_residual.to_bits());
                    if let Some(ref expected) = expected {
                        assert_eq!(&answer, expected);
                    } else {
                        expected = Some(answer);
                    }
                }
                if trial > 0 {
                    samples.push(elapsed / 8.0);
                }
            }
            samples.sort_by(f64::total_cmp);
            let answer = expected.unwrap();
            let mut fingerprint = std::hash::DefaultHasher::new();
            answer.hash(&mut fingerprint);
            println!(
                "dispatch-pcg threads={threads} rows={rows} ms={:.6} min={:.6} max={:.6} \
                 iterations={} answer={:016x}",
                (samples[3] + samples[4]) / 2.0, samples[0], samples[7],
                answer.1, fingerprint.finish(),
            );
        }
    }

    #[test]
    #[ignore = "release solve timing, run alone with --nocapture"]
    fn benchmark_pcg_scheduling() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            for rows in [81920, 200001, 600001] {
                let matrix = spd(rows);
                let n = 3 * rows as usize;
                let mut samples = [Vec::new(), Vec::new()];
                let mut expected = None;
                for trial in 0..7 {
                    for mode in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let mut device = crate::driver::launch::scheduling_tests::device(mode);
                        let resident = matrix.upload(&mut device);
                        let diagonal = staged_block(&mut device, &matrix.zero_diagonal, "test.zero");
                        let inverse = matrix.inverse(&mut device);
                        let b = staged_block(&mut device,
                            &(0..n).map(|i| (i % 17) as f32 / 17.0).collect::<Vec<_>>(), "test.b");
                        let zero = vec![0.0; n];
                        let mut x = staged_block(&mut device, &zero, "test.x");
                        let op = matrix.op(&resident, diagonal.handle());
                        let mut work = matrix.work(&mut device);
                        for warmup in [true, false] {
                            x.write(&mut device, 0, &zero).unwrap();
                            let start = std::time::Instant::now();
                            let report = solve(&mut device, &op, inverse.handle(), b.handle(),
                                x.handle(), &mut work, None, 1.0e-5, 100).unwrap();
                            let elapsed = start.elapsed().as_secs_f64() * 1e3;
                            assert_eq!(report.outcome, Outcome::Converged);
                            let value: Vec<u32> = read_back(&mut device, &x, n)
                                .into_iter().map(f32::to_bits).collect();
                            let answer = (value, report.iterations, report.relative_residual.to_bits());
                            if let Some(ref expected) = expected { assert_eq!(&answer, expected); }
                            else { expected = Some(answer); }
                            if !warmup {
                                samples[mode].push(elapsed);
                                println!("pcg rows={rows} trial={trial} mode={mode} iterations={} ms={elapsed:.3}",
                                    report.iterations);
                            }
                        }
                    }
                }
                for (mode, values) in samples.iter_mut().enumerate() {
                    values.sort_by(f64::total_cmp);
                    println!("pcg-summary rows={rows} mode={mode} ms={:.3} min={:.3} max={:.3}",
                        values[3], values[0], values[6]);
                }
            }
        });
    }

    /// A diagonally dominant SPD matrix in the fixed-CSR form, upper triangle
    /// stored with the transpose table naming the mirrored blocks.
    struct Spd {
        index: Vec<u32>,
        offset: Vec<u32>,
        value: Vec<f32>,
        transpose_pair: Vec<u32>,
        transpose_offset: Vec<u32>,
        /// The fixed matrix's own diagonal blocks, kept separately as the
        /// preconditioner's source.
        diagonal: Vec<f32>,
        /// The operator's `C` term. ZERO for this fixture, so the composed
        /// operator is exactly the fixed matrix and the tests below measure PCG
        /// rather than the composition. Adding an exact zero block is exact, so
        /// the composition is still the one the driver runs.
        zero_diagonal: Vec<f32>,
        rows: u32,
    }

    fn spd(rows: u32) -> Spd {
        let n = rows as usize;
        let mut index = Vec::new();
        let mut offset = vec![0u32];
        let mut value = Vec::new();
        let mut diagonal = vec![0.0f32; 9 * n];
        for i in 0..n {
            // Diagonal block: strongly dominant so the system is well
            // conditioned and CG converges quickly.
            let d = 8.0 + (i % 5) as f32;
            index.push(i as u32);
            let block = [d, 0.3, 0.1, 0.3, d + 1.0, 0.2, 0.1, 0.2, d + 2.0];
            value.extend_from_slice(&block);
            diagonal[9 * i..9 * i + 9].copy_from_slice(&block);
            if i + 1 < n {
                index.push(i as u32 + 1);
                let o = 0.75;
                value.extend_from_slice(&[o, 0.0, 0.0, 0.0, o, 0.0, 0.0, 0.0, o]);
            }
            offset.push(index.len() as u32);
        }
        let mut transpose_pair = Vec::new();
        let mut transpose_offset = vec![0u32];
        for j in 0..n {
            if j > 0 {
                let source = j - 1;
                let slot = offset[source] as usize + 1;
                transpose_pair.push(source as u32);
                transpose_pair.push(slot as u32);
            }
            transpose_offset.push((transpose_pair.len() / 2) as u32);
        }
        Spd {
            index,
            offset,
            value,
            transpose_pair,
            transpose_offset,
            zero_diagonal: vec![0.0f32; 9 * n],
            diagonal,
            rows,
        }
    }

    /// A solved vector, read back off its device allocation.
    fn read_back(device: &mut impl Device, buffer: &Buffer<f32>, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        buffer
            .read(device, 0, &mut out)
            .expect("the solved vector reads back");
        out
    }

    /// A device copy of one block array, which the solve now takes by handle.
    fn staged_block(
        device: &mut impl Device,
        host: &[f32],
        label: &'static str,
    ) -> Buffer<f32> {
        let mut buffer = Buffer::<f32>::none();
        buffer
            .size(device, host.len(), AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, host)
            .expect("the test upload succeeds");
        buffer
    }

    /// This fixture's five arrays, on the device the solve runs on.
    ///
    /// THE VIEW IS ALL HANDLES NOW, so the fixture has to own allocations
    /// rather than lend slices. One of these is built per test and lives as
    /// long as the device does, which is what keeps a handle resolvable.
    struct SpdDevice {
        index: Buffer<u32>,
        offset: Buffer<u32>,
        value: Buffer<f32>,
        transpose_pair: Buffer<u32>,
        transpose_offset: Buffer<u32>,
        rows: u32,
    }

    /// As [`staged_block`], for an index array.
    fn staged_index(device: &mut impl Device, host: &[u32], label: &'static str) -> Buffer<u32> {
        let mut buffer = Buffer::<u32>::none();
        buffer
            .size(device, host.len(), AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, host)
            .expect("the test upload succeeds");
        buffer
    }

    impl SpdDevice {
        fn view(&self) -> spmv::FixedCsrView {
            spmv::FixedCsrView {
                index: self.index.span(0, self.index.len()),
                offset: self.offset.span(0, self.offset.len()),
                value: self.value.span(0, self.value.len()),
                transpose_pair: self.transpose_pair.span(0, self.transpose_pair.len()),
                transpose_offset: self.transpose_offset.span(0, self.transpose_offset.len()),
                rows: self.rows,
            }
        }
    }

    impl Spd {
        /// Upload the five arrays and hand back the owner.
        fn upload(&self, device: &mut impl Device) -> SpdDevice {
            SpdDevice {
                index: staged_index(device, &self.index, "test.spd.index"),
                offset: staged_index(device, &self.offset, "test.spd.offset"),
                value: staged_block(device, &self.value, "test.spd.value"),
                transpose_pair: staged_index(device, &self.transpose_pair, "test.spd.tpair"),
                transpose_offset: staged_index(
                    device,
                    &self.transpose_offset,
                    "test.spd.toffset",
                ),
                rows: self.rows,
            }
        }

        /// The block-Jacobi inverse of this fixture's diagonal.
        ///
        /// Wraps the caller-owned-buffer form so a test reads as one line; the
        /// buffer belongs to the caller because the real one is a driver
        /// allocation made once at `initialize()`.
        fn inverse(&self, device: &mut impl Device) -> Buffer<f32> {
            let mut host = vec![0.0f32; 9 * self.rows as usize];
            assert!(
                build_block_jacobi(&self.diagonal, &mut host, self.rows),
                "the fixture's diagonal blocks are SPD and must invert"
            );
            staged_block(device, &host, "test.inverse")
        }


        /// A workspace sized for this fixture.
        /// A workspace sized ON THE CALLER'S DEVICE. The fold outputs are device
        /// allocations now, and a handle is only meaningful to the backend that
        /// opened its arena, so sizing here and solving on another instance
        /// would read an arena that was never opened.
        fn work(&self, device: &mut impl Device) -> Workspace {
            let mut work = Workspace::default();
            work.size_for(device, self.rows)
                .expect("the fixture workspace sizes");
            work
        }

        /// The operator over this fixture's matrix, with a caller-supplied zero
        /// block diagonal: the operator carries a HANDLE now, and a handle has
        /// to name an allocation on the device the solve will run on.
        fn op(&self, resident: &SpdDevice, diagonal: ppf_cts_compute::Handle) -> Operator {
            Operator {
                fixed_pattern: None, tet_factor:None,
                dynamic: None,
                fixed: resident.view(),
                diagonal,
            }
        }
    }


    /// A BREAKDOWN NAMES ITS CAUSE, because the two causes send a reader to
    /// different code.
    ///
    /// These two matter beyond tidiness. `step.rs` turns either into a fatal
    /// abort, and the whole `pAp<=0` triage keys on which one it is: an
    /// indefinite MATRIX is a lost PSD projection or a dropped CSR block in the
    /// assembly, while a non-SPD PRECONDITIONER is a NaN or infinite diagonal
    /// block. Reporting one message for both also collapses them onto the
    /// `CrashKind::Cg` an ordinary iteration-cap exhaustion produces, and the
    /// natural response to THAT is to raise `cg-max-iter`.
    #[test]
    fn an_indefinite_matrix_is_reported_as_the_matrix_and_not_the_preconditioner() {
        // NEGATE THE MATRIX AND NOTHING ELSE. The preconditioner is built from
        // `diagonal`, which is untouched, so it stays SPD and `r.z` is positive
        // at the seed: the only thing broken is the operator, and `p^T A p` is
        // then negative by the full magnitude of the SPD form rather than by a
        // round-off sliver, which is what puts it past the bound.
        let mut a = spd(200);
        for v in &mut a.value {
            *v = -*v;
        }
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 7) as f32 - 3.0) * 0.25).collect();
        let mut device = host_device();
        let b_in = staged_block(&mut device, &b_host, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);
        let mut x = staged_block(&mut device, &vec![0.0f32; n], "test.x");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let report = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-5, 500).expect("the solve returns a report rather than an error");
        assert_eq!(
            report.outcome,
            Outcome::IndefiniteMatrix,
            "a negated SPD operator is indefinite by the whole magnitude of its \
             own quadratic form, so this must name the MATRIX; got {report:?}"
        );
    }

    /// THE LATCH IS WHAT MAKES THE STRIDED RESIDUAL READ SAFE, so it is tested
    /// directly rather than through a solve.
    ///
    /// The host samples the residual every `RESID_CHECK_STRIDE` iterations
    /// while it is far from the tolerance, so a breakdown can fire on an
    /// iteration nobody looks at. The live `cause_out` slot is rewritten every
    /// iteration, so without a sticky slot a `pAp <= 0` followed by three
    /// healthy iterations would be ERASED before the next check and the solve
    /// would continue on a direction the guard had already rejected. That is
    /// the `pAp <= 0` family, which must abort loudly.
    ///
    /// Every existing breakdown test breaks at iteration 1, which is always a
    /// check, so none of them covers this.
    #[test]
    fn a_breakdown_between_two_checks_survives_to_the_next_one() {
        let mut device = host_device();
        let mut floats = ReadbackBuffer::<f32>::default();
        floats
            .size(&mut device, 5, AllocLabel("test.pcg_latch.f32"))
            .expect("the test allocation succeeds");
        // THE LATCHED VALUE GETS ITS OWN ALLOCATION, because `seed` writes a
        // WHOLE buffer: keeping it beside the inputs meant re-seeding the
        // inputs for the next dispatch silently cleared the latch, which is the
        // very thing under test.
        let mut latched = ReadbackBuffer::<f32>::default();
        latched
            .size(&mut device, 1, AllocLabel("test.pcg_latch.value"))
            .expect("the test allocation succeeds");
        latched
            .seed(&mut device, &[0.0])
            .expect("the latched value starts clear");
        // THE ANNOUNCEMENT FLAG, likewise its own allocation and for the same
        // reason: the probe skips reading the int latch when this reads zero,
        // so a flag cleared by a re-seed would hide the very latch under test.
        let mut fired = ReadbackBuffer::<f32>::default();
        fired
            .size(&mut device, 1, AllocLabel("test.pcg_latch.fired"))
            .expect("the test allocation succeeds");
        fired
            .seed(&mut device, &[0.0])
            .expect("the flag starts clear");
        let mut ints = ReadbackBuffer::<i32>::default();
        ints.size(&mut device, 4, AllocLabel("test.pcg_latch.i32"))
            .expect("the test allocation succeeds");

        // slots: 0 rz, 1 p_ap, 2 absolute_sum, 3 value, 4 noise, 5 break_value
        // ints:  0 cause_out, 1 break_cause, 2 break_iteration, 3 counter
        ints.seed(&mut device, &[PCG_BREAK_NONE; 4])
            .expect("the latch starts clear");

        let mut fire = |device: &mut crate::driver::launch::HostDevice,
                        p_ap: f32| {
            floats
                .seed(device, &[1.0, p_ap, 1.0e-30, 0.0, 0.0])
                .expect("the inputs upload");
            // NOTHING RE-SEEDS THE INT BLOCK, which holds both the sticky latch
            // this test exists for and the iteration counter. The counter opens
            // at zero and the body advances it, so the three calls below ARE
            // iterations 1, 2 and 3, which is the same advance the solve makes.
            let args = PcgAlphaResidentArgs {
                rz: floats.span(0, 1),
                p_ap: floats.span(1, 1),
                absolute_sum: floats.span(2, 1),
                iteration_counter: ints.span(3, 1),
                value_out: floats.span(3, 1),
                noise_out: floats.span(4, 1),
                cause_out: ints.span(0, 1),
                break_cause: ints.span(1, 1),
                break_value: latched.span(0, 1),
                break_iteration: ints.span(2, 1),
                break_fired: fired.span(0, 1),
                count: 1,
                seam_arena_count: 0,
            };
            device
                .run("test.pcg_latch", |encoder| {
                    // Safety: one element, three disjoint out-slots, and the
                    // two buffers are borrowed for the whole call.
                    unsafe { encoder.elements(&args, 1) }
                })
                .expect("the dispatch succeeds");
        };

        // Iteration 1 breaks: a curvature negative far beyond any round-off
        // bound, which is the fatal classification rather than the truncation.
        fire(&mut device, -1.0);
        // Iterations 2 and 3 are healthy and REWRITE the live cause slot, which
        // is exactly what the latch has to survive.
        fire(&mut device, 1.0);
        fire(&mut device, 1.0);

        ints.download(&mut device).expect("the latch reads back");
        latched.download(&mut device).expect("the value reads back");
        assert_eq!(
            ints.host()[0],
            PCG_BREAK_NONE,
            "the LIVE cause slot must show the last iteration's healthy verdict, \
             or this test is not exercising the overwrite it exists for"
        );
        assert_eq!(
            ints.host()[1],
            PCG_BREAK_PAP,
            "the STICKY cause must survive two healthy iterations; if it does \
             not, a strided residual read drops a pAp <= 0 and the solve runs \
             on a direction the guard rejected"
        );
        assert_eq!(
            ints.host()[2], 1,
            "the latch must name the iterate that BROKE, not the check that \
             noticed it, and the number is the DEVICE counter's: the first call \
             below is iteration 1 because the body advances the counter itself"
        );
        assert_eq!(
            latched.host()[0], -1.0,
            "the latched value must be the offending curvature, not a later \
             iterate's"
        );
        // THE FLAG THE PROBE GATES ON. If this were ever zero while the sticky
        // cause held a breakdown, the batch probe would skip the int readback
        // and the solve would run on a direction the guard had rejected, which
        // is the whole failure the gating could introduce.
        fired.download(&mut device).expect("the flag reads back");
        assert_ne!(
            fired.host()[0], 0.0,
            "the announcement flag must be set whenever the sticky cause is, or \
             the probe skips the readback that would have stopped the solve"
        );
    }

    /// A HEALTHY SOLVE LEAVES THE ANNOUNCEMENT FLAG CLEAR, which is the other
    /// half of what makes gating the int readback sound. The flag being
    /// NECESSARY is asserted above; this asserts it is not simply always set,
    /// which would make the gate a no-op that still passes that test.
    #[test]
    fn a_healthy_alpha_leaves_the_announcement_flag_clear() {
        let mut device = host_device();
        let mut floats = ReadbackBuffer::<f32>::default();
        floats
            .size(&mut device, 5, AllocLabel("test.pcg_clear.f32"))
            .expect("the test allocation succeeds");
        let mut latched = ReadbackBuffer::<f32>::default();
        latched
            .size(&mut device, 1, AllocLabel("test.pcg_clear.value"))
            .expect("the test allocation succeeds");
        latched.seed(&mut device, &[0.0]).expect("the value starts clear");
        let mut fired = ReadbackBuffer::<f32>::default();
        fired
            .size(&mut device, 1, AllocLabel("test.pcg_clear.fired"))
            .expect("the test allocation succeeds");
        fired.seed(&mut device, &[0.0]).expect("the flag starts clear");
        let mut ints = ReadbackBuffer::<i32>::default();
        // FOUR SLOTS: the live cause, the sticky one, the latched iteration,
        // and the iteration counter the body advances for itself.
        ints.size(&mut device, 4, AllocLabel("test.pcg_clear.i32"))
            .expect("the test allocation succeeds");
        ints.seed(&mut device, &[PCG_BREAK_NONE; 4])
            .expect("the latch starts clear");
        // A positive curvature far above the round-off bound: healthy on every
        // branch alpha classifies.
        floats
            .seed(&mut device, &[1.0, 1.0, 1.0e-30, 0.0, 0.0])
            .expect("the inputs upload");
        let args = PcgAlphaResidentArgs {
            rz: floats.span(0, 1),
            p_ap: floats.span(1, 1),
            absolute_sum: floats.span(2, 1),
            iteration_counter: ints.span(3, 1),
            value_out: floats.span(3, 1),
            noise_out: floats.span(4, 1),
            cause_out: ints.span(0, 1),
            break_cause: ints.span(1, 1),
            break_value: latched.span(0, 1),
            break_iteration: ints.span(2, 1),
            break_fired: fired.span(0, 1),
            count: 1,
            seam_arena_count: 0,
        };
        device
            .run("test.pcg_clear", |encoder| {
                // Safety: one element and every buffer is borrowed for the call.
                unsafe { encoder.elements(&args, 1) }
            })
            .expect("the dispatch succeeds");
        fired.download(&mut device).expect("the flag reads back");
        ints.download(&mut device).expect("the latch reads back");
        assert_eq!(
            ints.host()[1],
            PCG_BREAK_NONE,
            "the fixture must be healthy, or it is not testing the clear case"
        );
        assert_eq!(
            fired.host()[0], 0.0,
            "a healthy iteration must leave the flag clear, or gating the int \
             readback on it saves nothing"
        );
    }

    #[test]
    fn a_non_spd_preconditioner_is_reported_as_the_preconditioner() {
        // NEGATE THE PRECONDITIONER AND NOTHING ELSE, so `r^T M^-1 r` is
        // negative at the seed and the loop breaks before it ever forms
        // `p^T A p`. The operator is the untouched SPD fixture.
        let a = spd(200);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 7) as f32 - 3.0) * 0.25).collect();
        let mut device = host_device();
        let b_in = staged_block(&mut device, &b_host, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let mut inverse_host = vec![0.0f32; 9 * a.rows as usize];
        assert!(
            build_block_jacobi(&a.diagonal, &mut inverse_host, a.rows),
            "the fixture's diagonal blocks are SPD and must invert"
        );
        for v in &mut inverse_host {
            *v = -*v;
        }
        let inverse = staged_block(&mut device, &inverse_host, "test.negated_inverse");
        let mut x = staged_block(&mut device, &vec![0.0f32; n], "test.x");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let report = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-5, 500).expect("the solve returns a report rather than an error");
        assert_eq!(
            report.outcome,
            Outcome::NonSpdPreconditioner,
            "a negated block-Jacobi inverse makes r^T M^-1 r negative at the \
             seed, so this must name the PRECONDITIONER; got {report:?}"
        );
    }

    /// THE DEVICE FOLD IS THE HOST FOLD, up to the length at which the host
    /// fold stops having a second level.
    ///
    /// `super::reduce`'s contract is that the array is cut into fixed blocks,
    /// each summed in index order, and the block totals summed in index order.
    /// That is exactly two levels, so for any input the device fold finishes in
    /// two levels the two must agree BIT FOR BIT, not merely closely: the
    /// curvature bound this backend judges a solve against is stated over a
    /// fold of a particular shape, and a fold whose shape drifts silently
    /// invalidates it.
    ///
    /// The COOPERATIVE fold agrees with the element fold inside the fp32
    /// summation bound, and CANNOT agree to the bit.
    ///
    /// **THE BIT-IDENTICAL VERSION OF THIS TEST WAS WRITTEN FIRST AND IS
    /// UNACHIEVABLE.** The lanes take CONTIGUOUS runs in lane order and
    /// `compute::block_sum` folds them in ASCENDING order on this backend, which
    /// looked like it reproduced the element form's left-to-right sum. It does
    /// not: summing each run and then combining the run totals is
    /// `(a0+a1+a2+a3) + (a4+a5+a6+a7) + ...`, and fp32 addition is not
    /// associative, so it is not `((((a0+a1)+a2)+a3)+a4)+...`. **No lane
    /// decomposition reproduces a single thread's sum**, whatever order the
    /// lanes are combined in; only one lane doing all the work would.
    ///
    /// SO THE ASSERTION IS A BOUND, and it is `n * eps * sum|x|`, the standard
    /// fp32 summation bound and the one this solver already classifies against:
    /// `pcg_alpha` decides a breakdown by comparing a curvature against the
    /// round-off of its OWN absolute sum, because a cancelling sum's error is
    /// set by the magnitudes that went in and not by what came out. A bound
    /// taken relative to the TOTAL was tried and is wrong: on this input one
    /// block cancels from 1e7 terms down to 60, where five absolute is four ULP
    /// of the input and 8 percent of the answer.
    ///
    /// WHAT THIS MEANS FOR A CALLER. Moving a fold from the element form to this
    /// one MOVES ITS LAST BITS, so it is a numerics change and not a drop-in:
    /// `the_device_fold_agrees_with_the_host_fold_bit_for_bit` pins the PCG's
    /// fold against `reduce::sum`, and switching that recurrence over needs the
    /// host fold to take the same shape in the same change.
    #[test]
    fn the_cooperative_fold_agrees_with_the_element_fold_within_the_summation_bound() {
        use crate::driver::kernels::{VecBlockSumArgs, VecBlockSumCooperativeArgs};
        let mut device = host_device();
        // A CANCELLING INPUT, for the reason the fold test beside this one uses
        // one: an association only shows on a sum that cancels.
        let values: Vec<f32> = (0..5_000)
            .map(|i| {
                let magnitude = 1.0e6 * ((i % 17) as f32 + 1.0);
                if i % 2 == 0 { magnitude } else { -magnitude + 1.0 }
            })
            .collect();
        let source = staged_block(&mut device, &values, "test.coop.source");
        let blocks = values.len().div_ceil(FOLD_WIDTH);
        let mut element_out = Buffer::<f32>::none();
        element_out
            .size(&mut device, blocks, AllocLabel("test.coop.element"))
            .expect("the element output sizes");
        let mut group_out = Buffer::<f32>::none();
        group_out
            .size(&mut device, blocks, AllocLabel("test.coop.group"))
            .expect("the group output sizes");
        let source_handle = source.span(0, values.len());
        let element = VecBlockSumArgs {
            source: source_handle,
            length: values.len() as u32,
            width: FOLD_WIDTH as u32,
            total: element_out.handle(),
            count: blocks as u32,
            seam_arena_count: 0,
        };
        let cooperative = VecBlockSumCooperativeArgs {
            source: source_handle,
            length: values.len() as u32,
            width: FOLD_WIDTH as u32,
            total: group_out.handle(),
            count: blocks as u32,
            seam_arena_count: 0,
        };
        device
            .run("test.coop", |encoder| {
                // Safety: every buffer outlives the call.
                unsafe {
                    encoder.elements(&element, blocks as u32)?;
                    // THE WIDTH AND THE SCRATCH COME FROM THE DECLARATION, which
                    // is what stops a call site inventing a group-local array the
                    // body did not ask for.
                    encoder.groups(
                        &cooperative,
                        blocks as u32,
                        64,
                    )
                }
            })
            .expect("both folds dispatch");
        let mut from_elements = vec![0.0f32; blocks];
        let mut from_groups = vec![0.0f32; blocks];
        element_out
            .read(&mut device, 0, &mut from_elements)
            .expect("the element fold reads back");
        group_out
            .read(&mut device, 0, &mut from_groups)
            .expect("the group fold reads back");
        for (block, (a, b)) in
            from_elements.iter().zip(from_groups.iter()).enumerate()
        {
            // THE BOUND IS `n * eps * sum|x|`, WHICH IS THE STANDARD FP32
            // SUMMATION BOUND AND THE ONE THIS SOLVER ALREADY CLASSIFIES
            // AGAINST. `pcg_alpha` decides a breakdown by comparing a curvature
            // against the round-off of its OWN absolute sum, for exactly this
            // reason: a cancelling sum's error is set by the magnitudes that
            // went in, not by what came out. A bound taken relative to the
            // TOTAL would be a bound on the cancellation instead, and on this
            // input one block cancels from 1e7 terms down to 60, where five
            // absolute is four ULP of the input and 8 percent of the answer.
            let first = block * FOLD_WIDTH;
            let last = (first + FOLD_WIDTH).min(values.len());
            let magnitude: f32 =
                values[first..last].iter().map(|v| v.abs()).sum();
            let bound = (last - first) as f32 * f32::EPSILON * magnitude;
            assert!(
                (a - b).abs() <= bound,
                "block {block}: the cooperative fold is {a} against the \
                 element fold's {b}, {} apart against a summation bound of \
                 {bound}. Within the bound is the association difference this \
                 shape is allowed; past it is a different sum",
                (a - b).abs()
            );
        }
    }

    /// The input cancels, because a fold's shape only shows on one that does.
    #[test]
    fn the_device_fold_agrees_with_the_host_fold_bit_for_bit() {
        let mut device = host_device();
        // 40_000 values is 157 blocks, which the device fold reduces in two
        // levels, and it crosses the block boundary many times over.
        let values: Vec<f32> = (0..40_000)
            .map(|i| {
                let magnitude = 1.0e6 * ((i % 17) as f32 + 1.0);
                if i % 2 == 0 {
                    magnitude
                } else {
                    -magnitude + 1.0
                }
            })
            .collect();
        assert!(
            values.len() <= FOLD_WIDTH * FOLD_WIDTH,
            "past this length the device fold takes a third level and the host \
             fold does not, which is a different association by design"
        );
        let source = staged_block(&mut device, &values, "test.fold.source");
        let mut out = Buffer::<f32>::none();
        out.size(&mut device, 2, AllocLabel("test.fold.out"))
            .expect("the fold output sizes");
        let mut scratch = Buffer::<f32>::none();
        scratch
            .size(
                &mut device,
                fold_scratch_len(values.len()).max(1),
                AllocLabel("test.fold.scratch"),
            )
            .expect("the fold scratch sizes");
        let signed_out = out.span(0, 1);
        let magnitude_out = out.span(1, 1);
        let source_handle = source.span(0, values.len());
        device
            .run("test.fold", |encoder| {
                // Safety: every buffer outlives the call.
                unsafe {
                    encode_fold(
                        encoder,
                        source_handle,
                        values.len(),
                        &scratch,
                        signed_out,
                        false,
                    )?;
                    encode_fold(
                        encoder,
                        source_handle,
                        values.len(),
                        &scratch,
                        magnitude_out,
                        true,
                    )
                }
            })
            .expect("the fold dispatches");
        let mut got = [0.0f32; 2];
        out.read(&mut device, 0, &mut got).expect("the fold reads back");
        assert_eq!(
            got[0].to_bits(),
            reduce::sum(&values).to_bits(),
            "the device fold moved off the host fold: {} against {}",
            got[0],
            reduce::sum(&values)
        );
        assert_eq!(
            got[1].to_bits(),
            reduce::sum_abs(&values).to_bits(),
            "the device magnitude fold moved off the host one: {} against {}",
            got[1],
            reduce::sum_abs(&values)
        );
        assert!(
            reduce::sum_abs(&values) > reduce::sum(&values).abs(),
            "the fixture canceled too little to exercise what the shape is for"
        );
    }

    /// THE DUAL FOLD IS THE TWO CHAINS IT MERGES, BIT FOR BIT.
    ///
    /// The only claim [`encode_fold_dual`] makes is that sharing a dispatch
    /// changes no arithmetic: each source is summed over the same
    /// [`FOLD_WIDTH`] in the same ascending order at the same level as the
    /// single-array chain it replaces. This asserts that directly against
    /// [`encode_fold`], which is the thing a scene statistic cannot settle: the
    /// solver is non-deterministic run to run, so a displacement inside its
    /// envelope is consistent with a fold that is bit-exact AND with one that
    /// is merely close.
    ///
    /// THE LENGTHS ARE CHOSEN SO THE CHAINS HAVE DIFFERENT DEPTHS, because the
    /// level where one chain has reached its output and the other has not is
    /// the case the per-source counts exist for. A fixture with two equal
    /// lengths would exercise the equal-length pair and call it a pass.
    #[test]
    fn the_dual_fold_agrees_with_two_separate_folds_bit_for_bit() {
        // 300 reduces in two levels (300 -> 2 -> 1) and 70_000 in three
        // (70_000 -> 274 -> 2 -> 1), so each ordering puts a finished chain
        // beside a live one for a level.
        for (first_len, second_len) in
            [(300usize, 70_000usize), (70_000, 300), (40_000, 40_000), (1, 5_000)]
        {
            let mut device = host_device();
            let first: Vec<f32> = (0..first_len)
                .map(|i| {
                    let m = 1.0e6 * ((i % 17) as f32 + 1.0);
                    if i % 2 == 0 { m } else { -m + 1.0 }
                })
                .collect();
            let second: Vec<f32> = (0..second_len)
                .map(|i| {
                    let m = 1.0e5 * ((i % 13) as f32 + 1.0);
                    if i % 3 == 0 { -m } else { m + 0.5 }
                })
                .collect();
            let a = staged_block(&mut device, &first, "test.dual.a");
            let b = staged_block(&mut device, &second, "test.dual.b");
            let mut out = Buffer::<f32>::none();
            out.size(&mut device, 4, AllocLabel("test.dual.out"))
                .expect("the fold output sizes");
            let mut scratch = Buffer::<f32>::none();
            let first_span = fold_scratch_len(first_len).max(1);
            scratch
                .size(
                    &mut device,
                    first_span + fold_scratch_len(second_len).max(1),
                    AllocLabel("test.dual.scratch"),
                )
                .expect("the fold scratch sizes");
            let a_h = a.span(0, first_len);
            let b_h = b.span(0, second_len);
            device
                .run("test.dual", |encoder| {
                    // Safety: every buffer outlives the call.
                    unsafe {
                        encode_fold(encoder, a_h, first_len, &scratch, out.span(0, 1), false)?;
                        encode_fold(encoder, b_h, second_len, &scratch, out.span(1, 1), false)?;
                        encode_fold_dual(
                            encoder,
                            a_h,
                            first_len,
                            b_h,
                            second_len,
                            &scratch,
                            0,
                            first_span,
                            out.span(2, 1),
                            out.span(3, 1),
                        )
                    }
                })
                .expect("the folds dispatch");
            let mut got = [0.0f32; 4];
            out.read(&mut device, 0, &mut got).expect("the folds read back");
            assert_eq!(
                got[2].to_bits(),
                got[0].to_bits(),
                "the merged first chain moved off the separate one at \
                 ({first_len}, {second_len}): {} against {}",
                got[2],
                got[0]
            );
            assert_eq!(
                got[3].to_bits(),
                got[1].to_bits(),
                "the merged second chain moved off the separate one at \
                 ({first_len}, {second_len}): {} against {}",
                got[3],
                got[1]
            );
        }
    }

    /// SPLITTING THE MAGNITUDE LEVEL OFF AN L1 FOLD CHANGES NOTHING EITHER.
    ///
    /// The PCG dispatches the first level of `||r||_1` alone so the levels
    /// above it can share a chain with `r . z`. That is only sound if one
    /// magnitude level followed by plain levels is the same number as
    /// [`encode_fold`] with `absolute` set, which is what this asserts.
    #[test]
    fn the_split_magnitude_level_agrees_with_the_absolute_fold_bit_for_bit() {
        for count in [1usize, FOLD_WIDTH, 40_000, 70_000] {
            let mut device = host_device();
            let values: Vec<f32> = (0..count)
                .map(|i| {
                    let m = 1.0e6 * ((i % 17) as f32 + 1.0);
                    if i % 2 == 0 { m } else { -m + 1.0 }
                })
                .collect();
            let source = staged_block(&mut device, &values, "test.absplit.source");
            let tail = staged_block(&mut device, &[7.5f32], "test.absplit.tail");
            let mut out = Buffer::<f32>::none();
            out.size(&mut device, 3, AllocLabel("test.absplit.out"))
                .expect("the fold output sizes");
            let mut scratch = Buffer::<f32>::none();
            scratch
                .size(
                    &mut device,
                    2 * fold_scratch_len(count).max(1),
                    AllocLabel("test.absplit.scratch"),
                )
                .expect("the fold scratch sizes");
            let source_h = source.span(0, count);
            let tail_h = tail.span(0, 1);
            let tail_out = out.span(2, 1);
            device
                .run("test.absplit", |encoder| {
                    // Safety: every buffer outlives the call.
                    unsafe {
                        encode_fold(encoder, source_h, count, &scratch, out.span(0, 1), true)?;
                        let (rest, rest_len) = encode_fold_abs_level(
                            encoder,
                            source_h,
                            count,
                            &scratch,
                            out.span(1, 1),
                        )?;
                        if rest_len > 1 {
                            // THE PRODUCTION SHAPE: the levels above the
                            // magnitude one run as the FIRST chain of a dual,
                            // based past what that level wrote. The second
                            // chain here is one element, so it reaches its
                            // output immediately and every level after that
                            // must leave it alone.
                            encode_fold_dual(
                                encoder,
                                rest,
                                rest_len,
                                tail_h,
                                1,
                                &scratch,
                                rest_len,
                                fold_scratch_len(count).max(1),
                                out.span(1, 1),
                                tail_out,
                            )?;
                        }
                        Ok(())
                    }
                })
                .expect("the folds dispatch");
            let mut got = [0.0f32; 3];
            out.read(&mut device, 0, &mut got).expect("the folds read back");
            assert_eq!(
                got[1].to_bits(),
                got[0].to_bits(),
                "splitting the magnitude level moved the L1 norm at {count}: {} against {}",
                got[1],
                got[0]
            );
        }
    }

    /// THE FOLD SCRATCH COVERS EVERY LEVEL IT DISPATCHES, including the third
    /// one a long input takes.
    ///
    /// `Buffer::span` panics past the allocation, so an undersized scratch is
    /// loud rather than a wild write; this asserts the sizing arithmetic
    /// agrees with the loop that consumes it, which is the pair that can drift.
    #[test]
    fn the_fold_scratch_covers_every_level() {
        for count in [1usize, 2, FOLD_WIDTH, FOLD_WIDTH + 1, 40_000, 200_000, 3_000_000] {
            let mut needed = 0usize;
            let mut remaining = count;
            loop {
                let blocks = remaining.div_ceil(FOLD_WIDTH);
                if blocks == 1 {
                    break;
                }
                needed += blocks;
                remaining = blocks;
            }
            assert_eq!(
                fold_scratch_len(count),
                needed,
                "the scratch sizing and the fold's own level walk disagree at \
                 {count} values"
            );
        }
    }

    /// ONE REGION PER ITERATION, WHICH IS WHAT DEVICE RESIDENCY MEANS HERE.
    ///
    /// `Counters::syncs` is one per `Device::run`, so it counts host round
    /// trips. The recurrence this replaced took eight of them an iteration:
    /// the matvec and its fold, two more folds for the curvature scale, the
    /// alpha verdict, the two updates, the preconditioner and its fold, the
    /// beta verdict and the direction update, each a separate boundary because
    /// a host fold sat between them. Every one of those scalars is on the
    /// device now, so an iteration is one boundary and the seed is one more.
    ///
    /// THIS IS THE GATE THAT NOTICES A HOST READ COMING BACK. A fold moved back
    /// to the host cannot stay inside the region, because an `Encoder` cannot
    /// read; it has to split the iteration in two, and that shows up here as a
    /// second sync per iteration.
    #[test]
    fn an_iteration_costs_one_host_round_trip() {
        let a = spd(600);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 11) as f32 - 5.0) * 0.5).collect();
        let mut device = host_device();
        let b_in = staged_block(&mut device, &b_host, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);
        let x = staged_block(&mut device, &vec![0.0f32; n], "test.x");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        device.counters_reset();
        let report = solve(
            &mut device,
            &a.op(&resident, zero_diag.handle()),
            inverse.handle(),
            b_in.handle(),
            x.handle(),
            &mut work,
            None,
            1.0e-5,
            500,
        )
        .expect("solve");
        assert_eq!(report.outcome, Outcome::Converged);
        // THE PROPERTY IS ONE SYNC PER BATCH, NOT PER ITERATION, and the batch
        // is the residual stride. `be_encoder_submit` synchronizes at the end
        // of every `device.run`, so a submit per iteration is a device stall
        // per iteration; the loop therefore encodes a whole batch and reads the
        // probe once, on the submit's own synchronize.
        //
        // The schedule is a check after iteration 1, then
        // every `RESID_CHECK_STRIDE` until the residual comes within
        // `NEAR_TOL_FACTOR` of the tolerance, after which every iteration is
        // its own batch. So the count is bounded BELOW by the tightened tail
        // and ABOVE by one sync per iteration, and it must be strictly under
        // the latter or the batching has stopped working.
        let syncs = device.counters().syncs;
        let per_iteration = 1 + u64::from(report.iterations);
        assert!(
            syncs < per_iteration,
            "the solve took {syncs} host round trips over {} iterations, which is \
             one per iteration: the submit is not batched, and every \
             iteration is paying a full device synchronize",
            report.iterations
        );
        assert!(
            syncs >= 2,
            "the solve took {syncs} round trips, which is fewer than the seed \
             plus one check; a solve that never samples its residual cannot \
             have detected convergence"
        );
    }

    #[test]
    fn it_solves_an_spd_system_and_the_residual_is_real() {
        let a = spd(600);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 11) as f32 - 5.0) * 0.5).collect();
        let b = &b_host;
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let x_host = vec![0.0f32; n];
        let mut x = staged_block(&mut device, &x_host, "test.x");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let report = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-5, 500).expect("solve");
        assert_eq!(
            report.outcome,
            Outcome::Converged,
            "a well-conditioned SPD system must converge; got {report:?}"
        );

        // Verify independently: recompute the residual from the returned x
        // rather than trusting the loop's own bookkeeping.
        let mut ax = vec![0.0f32; n];
        let mut ax_d = staged_block(&mut device, &ax, "test.ax");
        // Safety: every handle names a live allocation on this device.
        unsafe {
            spmv::fixed_csr_spmv(
                &mut device,
                &resident.view(),
                x.span(0, n),
                ax_d.span(0, n),
                n,
            )
        }
        .expect("the fixed-pattern matvec dispatches");
        ax_d.read(&mut device, 0, &mut ax).expect("ax reads back");
        let x = read_back(&mut device, &x, n);
        let residual: f32 = reduce::sum_abs(
            &b.iter().zip(ax.iter()).map(|(p, q)| p - q).collect::<Vec<f32>>(),
        );
        let scale = reduce::sum_abs(&b);
        assert!(
            residual / scale < 1.0e-4,
            "the returned x does not satisfy the system: relative residual {}",
            residual / scale
        );
    }

    #[test]
    fn the_preconditioner_reduces_the_iteration_count() {
        // The point of a preconditioner, checked rather than assumed: the same
        // system with an identity preconditioner must take more iterations.
        let a = spd(600);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 7) as f32 - 3.0) * 0.25).collect();
        let b = &b_host;
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let x1_host = vec![0.0f32; n];
        let mut x1 = staged_block(&mut device, &x1_host, "test.x1");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let with = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x1.handle(), &mut work, None, 1.0e-6, 2000).expect("solve");

        let mut identity_host = vec![0.0f32; 9 * a.rows as usize];
        for i in 0..a.rows as usize {
            identity_host[9 * i] = 1.0;
            identity_host[9 * i + 4] = 1.0;
            identity_host[9 * i + 8] = 1.0;
        }
        let x2_host = vec![0.0f32; n];
        let mut x2 = staged_block(&mut device, &x2_host, "test.x2");
        // NO SECOND DEVICE. `b_in` and `x2` are handles into the arena the first
        // one opened, and a fresh backend resolves the same arena index against
        // its own empty table: the unpreconditioned solve saw a zero right-hand
        // side and reported convergence at iteration zero.
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let identity = staged_block(&mut device, &identity_host, "test.identity");
        let without = solve(&mut device, &a.op(&resident, zero_diag.handle()), identity.handle(), b_in.handle(), x2.handle(), &mut work, None, 1.0e-6, 2000).expect("solve");

        assert_eq!(with.outcome, Outcome::Converged);
        assert_eq!(without.outcome, Outcome::Converged);
        assert!(
            with.iterations < without.iterations,
            "block-Jacobi took {} iterations against {} unpreconditioned, so it \
             is not preconditioning anything",
            with.iterations,
            without.iterations
        );
    }

    #[test]
    fn the_solve_is_deterministic_across_thread_counts() {
        let a = spd(900);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 13) as f32 - 6.0) * 0.3).collect();
        let b = &b_host;
        // ONE DEVICE for the whole test: the inverse below is a handle into its
        // arena, so a nested block with its own backend could not resolve it.
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let reference = {
            let x_host = vec![0.0f32; n];
            let mut x = staged_block(&mut device, &x_host, "test.x");
            let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
            let r = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-6, 500).expect("solve");
            (read_back(&mut device, &x, n), r.iterations)
        };
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| {
                let x_host = vec![0.0f32; n];
                let mut x = staged_block(&mut device, &x_host, "test.x");
                let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
                let r = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-6, 500).expect("solve");
                (read_back(&mut device, &x, n), r.iterations)
            });
            assert_eq!(
                reference.1, got.1,
                "the iteration count moved at {threads} threads"
            );
            for (i, (p, q)) in reference.0.iter().zip(got.0.iter()).enumerate() {
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "x[{i}] moved at {threads} threads; the solve must be \
                     deterministic, which is what the fixed fold shape buys"
                );
            }
        }
    }

    #[test]
    fn the_seed_is_used_rather_than_discarded() {
        // Cap the work at one iteration and compare where each start lands. If
        // the solve discarded the seed and began from zero, both would land in
        // the same place; because it does not, the seeded start is far closer to
        // the answer after the same single step.
        let a = spd(300);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 9) as f32 - 4.0) * 0.2).collect();
        let b = &b_host;
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let answer_host = vec![0.0f32; n];
        let mut answer = staged_block(&mut device, &answer_host, "test.answer");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let converged = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), answer.handle(), &mut work, None, 1.0e-7, 2000).expect("solve");
        assert_eq!(converged.outcome, Outcome::Converged);

        // The converged answer, on the host, so the two single-step solves below
        // can be measured against it.
        let answer_values = read_back(&mut device, &answer, n);
        let distance = |x: &[f32]| -> f32 {
            let d: Vec<f32> = x
                .iter()
                .zip(answer_values.iter())
                .map(|(p, q)| p - q)
                .collect();
            reduce::sum_abs(&d)
        };

        // ONE DEVICE throughout: every buffer here is a handle into its arena.
        let from_zero_host = vec![0.0f32; n];
        let mut from_zero = staged_block(&mut device, &from_zero_host, "test.from_zero");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), from_zero.handle(), &mut work, None, 1.0e-12, 1).expect("solve");

        // Warm-started from the converged answer rather than from zero.
        let mut from_answer = staged_block(&mut device, &answer_values, "test.from_answer");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), from_answer.handle(), &mut work, None, 1.0e-12, 1).expect("solve");

        assert!(
            distance(&read_back(&mut device, &from_answer, n))
                < distance(&read_back(&mut device, &from_zero, n)) * 1.0e-3,
            "after one iteration the seeded start is {} from the answer and the \
             zero start is {}; if the seed were discarded these would match",
            distance(&read_back(&mut device, &from_answer, n)),
            distance(&read_back(&mut device, &from_zero, n))
        );
    }

    #[test]
    fn a_better_seed_does_not_buy_fewer_iterations() {
        // The consequence of the denominator rule, asserted so nobody "fixes"
        // it later by switching to ||b||. The tolerance is relative to the
        // SEEDED residual, so starting closer tightens the target in proportion
        // and the iteration count does not collapse. The same property holds on
        // the CUDA side: a warm start is a wash, and that is correct rather
        // than a plumbing defect.
        let a = spd(400);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 9) as f32 - 4.0) * 0.2).collect();
        let b = &b_host;
        // ONE DEVICE for the whole test: the inverse below is a handle into its
        // arena, so a nested block with its own backend could not resolve it.
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let answer_host = vec![0.0f32; n];
        let mut answer = staged_block(&mut device, &answer_host, "test.answer");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let cold = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), answer.handle(), &mut work, None, 1.0e-6, 2000).expect("solve");
        assert_eq!(cold.outcome, Outcome::Converged);

        // The warm start is a COPY of the converged answer, staged as its own
        // allocation so the cold solve's result is not overwritten.
        let answer_values = read_back(&mut device, &answer, n);
        let mut warm = staged_block(&mut device, &answer_values, "test.warm");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let warm_report = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), warm.handle(), &mut work, None, 1.0e-6, 2000).expect("solve");
        assert!(
            warm_report.iterations > 0,
            "a warm start that converged in zero iterations would mean the \
             residual was formed against ||b|| rather than against the seed"
        );
    }

    /// A numerical gate on the SHARED body, runnable without a GPU.
    ///
    /// Routing CUDA's `invert()` through `block_jacobi.kernel.cpp` moved the
    /// device image by 88 instructions, because the shared body returns the
    /// eigenvalues and a validity flag the inline version did not materialize.
    /// The arithmetic path is identical by inspection; this is the check that it
    /// is identical in behavior.
    ///
    /// WHAT IS ASSERTED, and what deliberately is not. For a well-conditioned
    /// block the result is a real inverse. For an extremely anisotropic one it is
    /// NOT, and that is the design rather than a defect: eigenvalues below about
    /// `eps * lambda_max` are unresolvable by `symm3x3`, so they are floored, and
    /// the header states that this "only affects preconditioning quality, never
    /// correctness". A preconditioner that is SPD but inexact costs iterations;
    /// PCG still converges. So the invariants that hold for EVERY block are
    /// symmetry and positive definiteness, and exactness is asserted only where
    /// the design offers it.
    ///
    /// Measured while writing this, and pre-existing rather than introduced: on
    /// `diag(1e6, 1, 1)` the reciprocal of the smallest eigenvalue comes back as
    /// 0.0105 rather than 1.0, so `symm3x3` has already lost that direction at an
    /// anisotropy of 1e6. The same arithmetic ran inline before this refactor, so
    /// CUDA has always done this.
    #[test]
    fn the_inverse_is_exact_where_the_design_promises_it() {
        let well_conditioned: [[f32; 9]; 2] = [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 0.5, 0.25, 0.5, 5.0, 0.75, 0.25, 0.75, 6.0],
        ];
        for (case, block) in well_conditioned.iter().enumerate() {
            let mut arr = [0.0f32; 9];
            assert!(
                build_block_jacobi(block, &mut arr, 1),
                "case {case}: an SPD block must invert"
            );
            for col in 0..3 {
                let mut e = [0.0f32; 3];
                e[col] = 1.0;
                let back = spmv::mat3_mul(block, &spmv::mat3_mul(&arr, &e));
                for row in 0..3 {
                    let want = if row == col { 1.0 } else { 0.0 };
                    assert!(
                        (back[row] - want).abs() < 1.0e-4,
                        "case {case}: M M^-1 is not the identity at ({row}, {col}): \
                         got {} want {want}",
                        back[row]
                    );
                }
            }
        }
    }

    /// Symmetry and positive definiteness hold for EVERY block, including the
    /// ill-conditioned drape case the header's comment is about.
    ///
    /// These are the invariants the PCG guards depend on: a non-SPD
    /// preconditioner corrupts the search direction, which is the failure the
    /// eigendecomposition was chosen over a cofactor inverse to prevent.
    #[test]
    fn the_inverse_is_symmetric_and_positive_definite_at_any_conditioning() {
        let cases: [[f32; 9]; 4] = [
            [4.0, 0.5, 0.25, 0.5, 5.0, 0.75, 0.25, 0.75, 6.0],
            [1.0e6, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            // The drape case: normal stiffness ~1e11 against a tangential floor
            // ~1e2, where a raw cofactor inverse returns a sign-flipped block.
            [1.0e11, 0.0, 0.0, 0.0, 1.0e2, 0.0, 0.0, 0.0, 1.0e2],
            [1.0e-3, 1.0e-4, 0.0, 1.0e-4, 1.0e-3, 0.0, 0.0, 0.0, 1.0e-3],
        ];
        for (case, block) in cases.iter().enumerate() {
            let mut arr = [0.0f32; 9];
            assert!(
                build_block_jacobi(block, &mut arr, 1),
                "case {case}: an SPD block must invert"
            );
            let scale = arr.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0e-30);

            for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
                assert!(
                    (arr[3 * b + a] - arr[3 * a + b]).abs() <= 1.0e-6 * scale,
                    "case {case}: the inverse is not symmetric at ({a}, {b})"
                );
            }

            // Positive definite: v^T M^-1 v > 0 for a spread of directions.
            for v in [
                [1.0f32, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 0.5],
            ] {
                let mv = spmv::mat3_mul(&arr, &v);
                let quadratic: f32 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
                assert!(
                    quadratic > 0.0,
                    "case {case}: v^T M^-1 v = {quadratic} for v = {v:?}; a \
                     non-SPD preconditioner corrupts the PCG search direction, \
                     which is what the eigendecomposition exists to prevent"
                );
            }
        }
    }

    #[test]
    fn a_non_spd_block_is_refused_rather_than_inverted() {
        // A zero diagonal block has lambda_max 0, which the shared body reports
        // as invalid. Returning a plausible inverse here would freeze that
        // vertex silently, because a zero preconditioner block trips no PCG
        // guard: it contributes exactly zero to r.z and pAp, never a negative.
        let rows = 4u32;
        let mut diagonal = vec![0.0f32; 9 * rows as usize];
        for i in 0..rows as usize - 1 {
            diagonal[9 * i] = 1.0;
            diagonal[9 * i + 4] = 1.0;
            diagonal[9 * i + 8] = 1.0;
        }
        let mut inverse = vec![0.0f32; 9 * rows as usize];
        assert!(
            !build_block_jacobi(&diagonal, &mut inverse, rows),
            "a zero diagonal block must be refused, not inverted"
        );
    }

    /// The refusal above, through the path the SOLVE actually takes.
    ///
    /// The test above exercises the shared body; this one exercises the
    /// dispatch, and the two are not the same walk: the driver does not invert
    /// row by row on the host, so a body that refuses correctly while the entry
    /// point drops the verdict would pass that test and ship a solve that
    /// silently froze a vertex. What is asserted here is the CHANNEL: a
    /// non-SPD block must come back as `Fault::Device`, counted, with the row
    /// in its payload.
    #[test]
    fn the_dispatched_inversion_reports_a_non_spd_block_through_the_diagnostic_lane() {
        use crate::driver::kernels::BlockJacobiInvertRowArgs;
        let rows = 4u32;
        let mut diagonal = vec![0.0f32; 9 * rows as usize];
        for i in 0..rows as usize - 1 {
            diagonal[9 * i] = 1.0;
            diagonal[9 * i + 4] = 1.0;
            diagonal[9 * i + 8] = 1.0;
        }
        let mut device = host_device();
        let diagonal_in = staged_block(&mut device, &diagonal, "test.bj_diagonal");
        let mut inverse_out = Buffer::<f32>::none();
        inverse_out
            .size(&mut device, 9 * rows as usize, AllocLabel("test.bj_inverse"))
            .expect("the test allocation succeeds");
        let args = BlockJacobiInvertRowArgs {
            diagonal: diagonal_in.handle(),
            inverse: inverse_out.handle(),
            count: rows,
            seam_arena_count: 0,
        };
        // Safety: both allocations outlive the dispatch and hold one block per row.
        let result = device.run("test.block_jacobi_invert", |encoder| unsafe {
            encoder.elements(&args, rows)
        });
        match result {
            Err(Fault::Device { diag, .. }) => {
                assert_eq!(diag.failures, 1, "exactly the one zero block is refused");
                let first = diag.first.expect("the refusal carries its payload");
                assert_eq!(
                    first.payload[3], (rows - 1) as f32,
                    "the payload names the row that was refused, got {first}"
                );
            }
            other => panic!("a zero diagonal block must be refused, got {other:?}"),
        }
    }
}

/// The extra vectors the locked solve needs beside [`Workspace`].
///
/// SEPARATE FROM `Workspace` so an unlocked scene allocates none of it. Every
/// buffer is sized once for the run and reused, as the rest of the driver's are.
#[derive(Default)]
pub struct LockedWorkspace {
    /// The affine feasible correction the projector builds, and the tangent
    /// increment the recurrence solves for on top of it. `x = q + zsol`.
    pub q: Buffer<f32>,
    pub zsol: Buffer<f32>,
}

impl LockedWorkspace {
    pub fn allocate<D: Device>(&mut self, device: &mut D, rows: u32) -> Result<(), Fault> {
        let n = 3 * rows as usize;
        self.q.size(device, n, ppf_cts_compute::AllocLabel("pcg.locked.q"))?;
        self.zsol
            .size(device, n, ppf_cts_compute::AllocLabel("pcg.locked.zsol"))?;
        Ok(())
    }
}

/// PCG under an aggregate lock.
///
/// THE SYSTEM IS NOT THE UNLOCKED ONE WITH EXTRA STEPS. Writing the Newton
/// system as `M dx = b` and splitting `dx = q + z` with `q` the affine feasible
/// correction the projector builds and `z` a tangent increment, the equation
/// this solves is `Q M Q z = Q (b - M q)`. So it seeds from `q`, measures its
/// residual against `||Q (b - M q)||`, and returns `q + zsol` rather than the
/// iterate directly. That is also what preserves the seeded-residual denominator
/// rule: `err0` is the CONSTRAINED residual after the exact known correction,
/// never `||b||`.
///
/// EVERY PROJECTION IS ITS OWN BOUNDARY, and that is why this is the
/// host-syncing solve rather than the device-resident graph-captured one: a
/// projection is three dispatches with a host readback between the frames and
/// the rows, so it cannot be folded into a neighboring encoder region.
///
/// # Safety
/// Every handle must name a live allocation sized for the scene, and the
/// projector must have been `prepare`d against the same positions this solve
/// runs at, with `locked.q` holding the correction that call produced.
#[allow(clippy::too_many_arguments)]
pub unsafe fn solve_locked<D: Device>(
    device: &mut D,
    op: &Operator,
    projector: &mut super::lock::Projector<'_>,
    inverse_diagonal: ppf_cts_compute::Handle,
    b: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    work: &mut Workspace,
    locked: &mut LockedWorkspace,
    // THE PRECONDITIONER, exactly as the unlocked `solve` takes it.
    //
    // The caller builds the hierarchy BEFORE it branches on whether the scene is
    // aggregate-locked and hands it to both arms, so a locked scene gets the
    // preconditioner it asked for. A
    // locked solve that took block-Jacobi regardless would answer
    // `precond = schwarz` with a label rather than a preconditioner, which is
    // the failure this seam is most exposed to: a preconditioner that is
    // compiled, correct and reached by nothing.
    mut schwarz: Option<&mut super::schwarz::State>,
    tolerance: f32,
    max_iterations: u32,
) -> super::scene::FatalResult<Report> {
    let rows = op.rows();
    let n = 3 * rows as usize;
    // `q` IS AN INPUT, already built by the caller's `Projector::prepare`. The
    // dispatcher prepares and then calls the solve, so the affine correction is
    // formed once against the positions the frames were built at rather than
    // rebuilt inside the loop.
    let q = locked.q.handle();
    let zsol = locked.zsol.handle();

    let Workspace {
        ap,
        r,
        z,
        p,
        product,
        magnitude,
        row_absolute,
        row_curvature,
        verdict,
        verdict_cause,
        // THE FOLD SCRATCH AND THE SCALAR SLOTS, so this path reduces on the
        // DEVICE as the unlocked one does. Downloading the whole vector and
        // folding it here would be a relocation of the computation rather than
        // transport of a result, which is what rule (1a-0) forbids.
        //
        // NAMED `scalars` BECAUSE `scalar` IS TAKEN further down, by the alpha
        // verdict's `let scalar = pcg_alpha(...)`. A second `scalar` here would
        // be shadowed from that line on and the slots would quietly stop being
        // reachable half way through the loop.
        fold,
        scalar: scalars,
        ..
    } = work;
    let mut terms = DotTerms { product, magnitude };
    // THE APPLY WRITES ITS MAGNITUDE SUM SOMEWHERE AND THIS PATH IGNORES IT.
    // The locked and reduced solves take their curvature bound from the
    // Cauchy-Schwarz surrogate instead, because the operator they call is
    // wrapped in projections and its per-row sum is not the sum that forms
    // their `p^T A p`.
    let discard_absolute = row_absolute.handle();
    let discard_curvature = row_curvature.handle();

    // r = Q (b - M q), zsol = 0.
    op.apply(device, q, ap.handle(), discard_absolute, discard_curvature)?;
    combine(device, b, ap.handle(), r.handle(), 1.0, -1.0, n)?;
    projector
        .project(device, r.handle(), 1)?;
    combine(device, zsol, zsol, zsol, 0.0, 0.0, n)?;

    let err_slot = scalars.span(slot::ERR, 1);
    device.run("pcg.locked.err0", |encoder| {
        // Safety: `r` and the fold scratch are borrowed for the whole call.
        unsafe { encode_fold(encoder, r.handle(), n, fold, err_slot, true) }
    })?;
    scalars.download(device)?;
    let err0 = scalars.host()[slot::ERR];
    if err0 == 0.0 {
        // Already feasible and already solved: the answer is q itself.
        combine(device, q, q, x, 1.0, 0.0, n)?;
        return Ok(Report {
            outcome: Outcome::Converged,
            iterations: 0,
            relative_residual: 0.0,
        });
    }

    device.run("pcg.locked.precondition", |encoder| {
        // Safety: every buffer is borrowed for the whole call.
        unsafe {
            match schwarz.as_deref_mut() {
                Some(state) => super::schwarz::encode_whole_sweep(
                    encoder, state, r.handle(), z.handle(), rows),
                None => encode_preconditioner(
                    encoder, inverse_diagonal, r.handle(), z.handle(), rows),
            }
        }
    })?;
    projector
        .project(device, z.handle(), 1)?;
    let mut rz = terms.dot(device, r.handle(), z.handle(), n, fold, scalars)?;
    // The negation traps a NaN, which `<= 0.0` would let through; the idiom is
    // the unlocked solve's.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(rz > 0.0) {
        return Ok(Report {
            outcome: Outcome::NonSpdPreconditioner,
            iterations: 0,
            relative_residual: 1.0,
        });
    }
    combine(device, z.handle(), p.handle(), p.handle(), 1.0, 0.0, n)?;
    projector
        .project(device, p.handle(), 1)?;

    // `x = q + zsol`, which every exit below takes.
    let finish = |device: &mut D, zsol: ppf_cts_compute::Handle| -> Result<(), Fault> {
        combine(device, q, zsol, x, 1.0, 1.0, n)
    };


    for iteration in 1..=max_iterations {
        // Ap = Q M Q p. `p` is projected after every update, and projecting it
        // again here makes the identity explicit and limits accumulated fp32
        // drift.
        projector
            .project(device, p.handle(), 1)?;
        op.apply(device, p.handle(), ap.handle(), discard_absolute, discard_curvature)?;
        projector
            .project(device, ap.handle(), 1)?;

        device.run("pcg.locked.terms", |encoder| {
            // Safety: `p` and `ap` are borrowed for the whole call.
            unsafe { terms.encode_fill(encoder, p.handle(), ap.handle(), n) }
        })?;
        let pap_slot = scalars.span(slot::P_AP, 1);
        let product_h = terms.product.handle();
        device.run("pcg.locked.pap", |encoder| {
            // Safety: the product vector and the scratch are borrowed for the
            // whole call.
            unsafe {
                encode_fold(encoder, product_h, rows as usize, fold, pap_slot, false)
            }
        })?;
        scalars.download(device)?;
        let pap = scalars.host()[slot::P_AP];
        // As the unlocked solve: `|p|_2 |Ap|_2`, read after `pap` because it
        // reuses the same fold buffers.
        let absolute_dot =
            terms.norm_product(device, p.handle(), ap.handle(), n, fold, scalars)?;
        let scalar = pcg_alpha(device, verdict, verdict_cause, rz, pap, absolute_dot)?;
        if scalar.cause != PCG_BREAK_NONE {
            let outcome = if scalar.cause == PCG_BREAK_PAP {
                Outcome::IndefiniteMatrix
            } else {
                // Curvature below the round-off of its own reduction: truncate
                // Steihaug-style with the direction fp32 can still see, and
                // report it as a SUCCESS, since the direction so far is still a
                // descent direction.
                projector
                    .project(device, zsol, 3)?;
                finish(device, zsol)?;
                let reresid = residual_of(device, r, err0)?;
                // As on the unlocked path above, and for the same reason: a
                // truncation that returns success and is never mentioned makes
                // a step that stopped early read exactly like one that
                // converged.
                ::log::info!(
                    "* cg truncated: curvature within round-off of its own \
                     reduction at iter {} (reresid {:.3e})",
                    iteration,
                    reresid
                );
                return Ok(Report {
                    outcome: Outcome::CurvatureTruncated,
                    iterations: iteration,
                    relative_residual: reresid,
                });
            };
            return Ok(Report {
                outcome,
                iterations: iteration,
                relative_residual: 1.0,
            });
        }

        let alpha = scalar.value;
        device.run("pcg.locked.update", |encoder| {
            // Safety: `p`, `zsol`, `ap` and `r` are borrowed for the whole call
            // and the two updates touch disjoint destinations.
            unsafe {
                encode_add_scaled(encoder, p.handle(), zsol, alpha, n)?;
                encode_add_scaled(encoder, ap.handle(), r.handle(), -alpha, n)
            }
        })?;
        projector
            .project(device, r.handle(), 1)?;

        let resid_slot = scalars.span(slot::ERR, 1);
        device.run("pcg.locked.residual", |encoder| {
            // Safety: `r` and the fold scratch are borrowed for the whole call.
            unsafe { encode_fold(encoder, r.handle(), n, fold, resid_slot, true) }
        })?;
        scalars.download(device)?;
        let mut residual = scalars.host()[slot::ERR] / err0;
        let mut restart_from_true = false;
        if residual < tolerance {
            projector
                .project(device, zsol, 3)?;
            finish(device, zsol)?;
            // Probe the EXACT returned affine iterate q + projected(zsol),
            // not zsol alone and not a newly projected approximation of x.
            let recursive_residual = residual;
            op.apply(device, x, ap.handle(), discard_absolute, discard_curvature)?;
            combine(device, b, ap.handle(), r.handle(), 1.0, -1.0, n)?;
            projector.project(device, r.handle(), 1)?;
            device.run("pcg.locked.true_residual", |encoder| {
                unsafe { encode_fold(encoder, r.handle(), n, fold, resid_slot, true) }
            })?;
            scalars.download(device)?;
            residual = scalars.host()[slot::ERR] / err0;
            if residual < tolerance {
                // x already holds the verified result. Reprojecting zsol or
                // recombining it here could change its floating-point value.
                return Ok(Report {
                    outcome: Outcome::Converged,
                    iterations: iteration,
                    relative_residual: residual,
                });
            }
            ::log::info!("pcg locked true-residual restart after iteration {}: recursive={:.9} true={:.9}",
                iteration,recursive_residual,residual);
            restart_from_true = true;
        }
        if iteration >= max_iterations || !residual.is_finite() {
            if !restart_from_true {
                projector
                    .project(device, zsol, 3)?;
                finish(device, zsol)?;
            }
            return Ok(Report {
                outcome: Outcome::MaxIterations,
                iterations: iteration,
                relative_residual: residual,
            });
        }

        device.run("pcg.locked.precondition", |encoder| {
            // Safety: as above.
            unsafe {
                match schwarz.as_deref_mut() {
                    Some(state) => super::schwarz::encode_whole_sweep(
                        encoder, state, r.handle(), z.handle(), rows),
                    None => encode_preconditioner(
                        encoder, inverse_diagonal, r.handle(), z.handle(), rows),
                }
            }
        })?;
        projector
            .project(device, z.handle(), 1)?;
        let rz_next = terms.dot(device, r.handle(), z.handle(), n, fold, scalars)?;
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(rz_next > 0.0) {
            return Ok(Report {
                outcome: Outcome::NonSpdPreconditioner,
                iterations: iteration,
                relative_residual: residual,
            });
        }
        // A rejected convergence starts a fresh direction using true r and
        // the original fixed M; all other iterations retain their recurrence.
        let beta = if restart_from_true { 0.0 } else { rz_next / rz };
        combine(device, z.handle(), p.handle(), p.handle(), 1.0, beta, n)?;
        projector
            .project(device, p.handle(), 1)?;
        rz = rz_next;
    }

    projector
        .project(device, zsol, 3)?;
    finish(device, zsol)?;
    Ok(Report {
        outcome: Outcome::MaxIterations,
        iterations: max_iterations,
        relative_residual: 1.0,
    })
}

#[path="warm_start_policy.rs"]
mod warm_start_policy;
#[path="lanczos_trace.rs"]
mod lanczos_trace;
#[path = "factor_refresh_policy.rs"]
mod factor_refresh_policy;
use warm_start_policy::{WarmKey, ColdCost};

/// A prefix needs warm-up and enough remaining interior iterations to amortize
/// graph setup. A borrowed factor keeps its existing wall-time refresh policy.
fn stride_graph_capture_candidate(iteration: u32, cap: u32, borrowed_ms: Option<(f64, f64)>) -> bool {
    iteration == 5 && cap.saturating_sub(iteration) >= 4
        && borrowed_ms.is_none_or(|(spent, allowed)| 2.0 * spent < allowed)
}

#[cfg(test)]
mod stride_graph_capture_tests {
    use super::stride_graph_capture_candidate as candidate;

    #[test]
    fn only_captures_a_warm_interior_iteration_with_headroom() {
        assert!(!candidate(1, 100, None));
        assert!(!candidate(5, 8, None));
        assert!(candidate(5, 9, None));
        assert!(!candidate(9, 9, None));
        assert!(candidate(5, 50, Some((0.25, 1.0))));
        assert!(!candidate(5, 50, Some((0.5, 1.0))));
        assert!(!candidate(5, 50, Some((f64::NAN, 1.0))));
    }
}

/// The vectors the PDRD reduced solve needs beside [`Workspace`].
///
/// SIZED ONCE FOR THE RUN and reused by every call, which is the rule the
/// driver is held to: `xv` and `mxv` in particular are shared by every
/// application of the reduced operator within one solve, so they are per-solve
/// state rather than per-call temporaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FactorDomainKey {
    device_instance: u64,
    workspace_generation: u64,
    nrow: usize,
    bodies: usize,
    body_base: usize,
    dim: usize,
    roots: Option<super::roots::ConstraintDomain>,
}

struct CachedBodyFactor {
    owner: super::body_factor_device::BodyFactorDevice,
    key: FactorDomainKey,
    prepare_ms: f64,
    iteration_debt: Option<FactorIterationDebt>,
}

/// Opt-in cache replacement heuristic. Charge only completed same-dt
/// verified solves. One fresh segment's iteration count is the allowance for
/// extra iterations, independent of the first factor's wall-clock setup.
#[derive(Clone, Copy)]
struct FactorIterationDebt {
    dt_bits: u32,
    fresh_iterations: u32,
    excess_iterations: u32,
}

impl FactorIterationDebt {
    fn charge(mut self, borrowed_iterations: u32) -> Self {
        self.excess_iterations = self.excess_iterations.saturating_add(
            borrowed_iterations.saturating_sub(self.fresh_iterations),
        );
        self
    }

    fn expired(self) -> bool {
        self.excess_iterations >= self.fresh_iterations
    }
}

#[derive(Default)]
pub struct RigidWorkspace {
    full_body_reuse: full_body_native::Reuse,
    full_body_symbolic: full_body_factor::SymbolicCache,
    full_body_binding_generation: u64,
    body_inverse: Option<super::body_inverse_device::BodyInverseDevice>,
    body_factor: Option<super::body_factor_device::BodyFactorDevice>,
    body_factor_cache: Option<CachedBodyFactor>,
    factor_workspace_generation: u64,
    active_factor_key: Option<FactorDomainKey>,
    active_factor_prepare_ms: f64,
    active_factor_borrowed: bool,
    active_factor_iteration_debt: Option<FactorIterationDebt>,
    active_factor_fresh_prefix: Option<u32>,
    body_host_tree: Option<HostBodyTree>,
    body_inverse_debug: Option<BodyTreeDebug>,
    body_group_reference: Option<BodyGroupReference>,
    body_seed_x: Buffer<f32>,
    body_seed_r: Buffer<f32>,
    warm_history: Buffer<f32>,
    warm_history2: Buffer<f32>,
    warm_ad1: Buffer<f32>,
    warm_ad2: Buffer<f32>,
    warm_seed: Buffer<f32>,
    warm_free_rows: Buffer<u32>,
    warm_key: Option<WarmKey>,
    warm_count: u8,
    warm_cold_cost: ColdCost,
    /// Borrowed per-frozen-solve blocks for optional failure diagnostics only.
    /// The step caller keeps this allocation alive throughout the solve.
    pub failure_precond_blocks: Option<ppf_cts_compute::Handle>,
    /// Reduced, `dim` floats each.
    pub f: Buffer<f32>,
    pub xr: Buffer<f32>,
    pub r: ReadbackBuffer<f32>,
    pub z: Buffer<f32>,
    pub p: Buffer<f32>,
    pub rp: Buffer<f32>,
    /// Full space, `3 * nrow` floats each: the operator's own scratch.
    pub xv: Buffer<f32>,
    pub mxv: Buffer<f32>,
    /// One `vec![0.0; ...]` the reduction's clears write from, so no clear
    /// allocates.
    pub zero: Vec<f32>,
    /// The per-group L1 norms, seeded and current.
    /// The per-group L1 norms as the DEVICE writes them: slot 0 is the cloth
    /// block and slots 1.. are the bodies. Only these `1 + bodies` floats cross
    /// the seam; the reduced vector itself stays on the device.
    pub group_norm: ReadbackBuffer<f32>,
    pub group_initial_device: Buffer<f32>,
    pub group_initial: Vec<f32>,
    pub group_current: Vec<f32>,
    /// THE REDUCED SOLVE NEEDS ITS OWN DOT TERMS, sized to `dim` rather than to
    /// `3 * rows`. `DotTerms::fill` takes the extent from its own buffer's
    /// length and ignores the `n` it is passed, so the main workspace's terms
    /// would fold over the FULL vertex length and read past the end of every
    /// reduced vector. That is not a stale-tail question: the products
    /// themselves would be computed from whatever lies beyond `dim`.
    pub product: ReadbackBuffer<f32>,
    pub magnitude: ReadbackBuffer<f32>,
}

impl RigidWorkspace {
    /// Release before replacing the workspace or its device-owned arena.
    pub(crate) fn release_factor_cache<D: Device>(&mut self, device: &mut D) -> Result<(), Fault> {
        self.full_body_symbolic = full_body_factor::SymbolicCache::default();
        if let Some(mut cached) = self.body_factor_cache.take() {
            cached.owner.free(device)?;
        }
        Ok(())
    }
    fn begin_warm<D:Device>(&mut self, device:&mut D, dim:usize, mode:u8) -> Result<(Option<WarmKey>,u8),Fault> {
        let prior = self.warm_key.take();
        let count = std::mem::take(&mut self.warm_count);
        if mode != 0 {
            self.warm_history.size(device, dim, ppf_cts_compute::AllocLabel("pcg.warm.history"))?;
            self.warm_seed.size(device, dim, ppf_cts_compute::AllocLabel("pcg.warm.seed"))?;
            self.warm_free_rows.size(device, dim, ppf_cts_compute::AllocLabel("pcg.warm.free_rows"))?;
            if mode == 2 {
                self.warm_history2.size(device, dim, ppf_cts_compute::AllocLabel("pcg.warm.history2"))?;
                self.warm_ad1.size(device, dim, ppf_cts_compute::AllocLabel("pcg.warm.ad1"))?;
                self.warm_ad2.size(device, dim, ppf_cts_compute::AllocLabel("pcg.warm.ad2"))?;
            }
        }
        Ok((prior,count))
    }
    pub(crate) fn invalidate_full_body_binding(&mut self) {
        self.full_body_reuse = full_body_native::Reuse::default();
        self.full_body_symbolic = full_body_factor::SymbolicCache::default();
        // The old entry is dropped before replacing its binding.
        self.full_body_binding_generation = self.full_body_binding_generation.saturating_add(1);
    }

    pub fn allocate<D: Device>(
        &mut self,
        device: &mut D,
        dim: usize,
        rows: usize,
        bodies: usize,
    ) -> Result<(), Fault> {
        use ppf_cts_compute::AllocLabel;
        self.full_body_reuse = full_body_native::Reuse::default();
        self.release_factor_cache(device)?;
        self.factor_workspace_generation = self.factor_workspace_generation.checked_add(1)
            .ok_or_else(|| Fault::Platform { call: "factor cache", detail: "workspace generation exhausted".into() })?;
        self.f.size(device, dim, AllocLabel("pdrd.cg.f"))?;
        self.xr.size(device, dim, AllocLabel("pdrd.cg.xr"))?;
        self.r.size(device, dim, AllocLabel("pdrd.cg.r"))?;
        self.z.size(device, dim, AllocLabel("pdrd.cg.z"))?;
        self.p.size(device, dim, AllocLabel("pdrd.cg.p"))?;
        self.rp.size(device, dim, AllocLabel("pdrd.cg.rp"))?;
        self.xv.size(device, 3 * rows, AllocLabel("pdrd.cg.xv"))?;
        self.mxv.size(device, 3 * rows, AllocLabel("pdrd.cg.mxv"))?;
        // SIZED IN ROWS OF THREE, NOT IN FLOATS. `DotTerms::fill` takes its
        // extent from this buffer's own length and each thread folds THREE
        // components, which is why the main workspace sizes its pair by the
        // VERTEX count. Sizing them by `dim` makes every fold read three times
        // past the end of a reduced vector, and the products come back too
        // large by a factor that grows as the solve proceeds. `dim` is
        // `3 * n_cloth + 6 * n_bodies` and so is always a multiple of three.
        debug_assert_eq!(dim % 3, 0, "the reduced dimension is a multiple of three");
        self.product
            .size(device, dim / 3, ppf_cts_compute::AllocLabel("pdrd.cg.product"))?;
        self.magnitude
            .size(device, dim / 3, ppf_cts_compute::AllocLabel("pdrd.cg.magnitude"))?;
        self.zero = vec![0.0; dim.max(3 * rows).max(36 * bodies)];
        self.group_norm.size(
            device,
            3 + bodies, // The final slot optionally carries r.z in the same readback.
            ppf_cts_compute::AllocLabel("pdrd.cg.group_norm"),
        )?;
        self.group_initial_device.size(device, 1+bodies, AllocLabel("pdrd.cg.group_initial"))?;
        self.group_initial = vec![0.0; 1 + bodies];
        self.group_current = vec![0.0; 1 + bodies];
        Ok(())
    }
}

/// PCG on the PDRD reduced system.
///
/// IT SOLVES `R u = P^T b` WITH `R = P^T M P`, and `R` is never formed: each
/// application prolongs the reduced vector to the full one, applies the
/// assembled per-vertex Newton operator, restricts the result back, and
/// projects each body's six DOFs onto its joint and lock subspace.
///
/// THE TOLERANCE IS MEASURED PER DOF GROUP, which is the property this solve
/// most easily loses and the reason `super::pdrd::worst_relative_residual`
/// exists. Collapsing it to one norm over the reduced vector lets a heavy body,
/// whose 6x6 is exactly preconditioned and dies in one step, carry the whole
/// residual across the tolerance while the cloth is untouched.
///
/// IT IS SEEDED FROM THE CALLER'S GUESS rather than zeroed, and through
/// `seed_restrict` rather than the force restriction. Every prescribed row of
/// `x` carries its exact Dirichlet correction, and the whole system routes
/// through here as soon as any PDRD body exists; a memset would throw that seed
/// away, leaving a prescribed row converging only to PCG tolerance and putting
/// its displacement into `err0` beside genuine force rows, which is the
/// pin-dominated denominator that must never be the tolerance's scale.
///
/// # Safety
/// Every handle must name a live allocation sized for the scene, the reduction
/// must have been given this iteration's rotated rest vectors, and the
/// preconditioner must have been built against the matrices `op` applies.
#[allow(clippy::too_many_arguments)]
/// The per-group L1 norms of `reduced`, FOLDED ON THE DEVICE.
///
/// Three dispatches form them: an `fabsf` map over the cloth rows, a device
/// reduction of that into slot 0, and one thread per body summing its own six
/// rows into slot `1 + b`. Only the `1 + bodies` results cross the seam.
/// Downloading the whole reduced vector and folding it on the host instead
/// would relocate the computation rather than transport its result, which is
/// what rule (1a-0) forbids.
///
/// THE CLOTH FOLD IS SKIPPED WHEN THERE IS NO CLOTH BLOCK and a zero is written
/// instead: `encode_fold` has no meaning over zero rows.
unsafe fn group_l1_device<D: Device>(
    device: &mut D,
    reduced: ppf_cts_compute::Handle,
    norms: &mut ReadbackBuffer<f32>,
    fold: &Buffer<f32>,
    body_base: usize,
    bodies: usize,
    out: &mut [f32],
) -> Result<(), Fault> {
    group_l1_after(device, reduced, norms, fold, body_base, bodies, out, |_| Ok(()))
}

/// Append work that produces `reduced` before measuring its DOF groups in
/// the same ordered region. The group measurement and readback stay the same.
unsafe fn group_l1_after<D: Device>(
    device: &mut D,
    reduced: ppf_cts_compute::Handle,
    norms: &mut ReadbackBuffer<f32>,
    fold: &Buffer<f32>,
    body_base: usize,
    bodies: usize,
    out: &mut [f32],
    produce: impl FnOnce(&mut dyn Encoder) -> Result<(), Fault>,
) -> Result<(), Fault> {
    debug_assert_eq!(out.len(), 1 + bodies);
    let cloth_slot = norms.span(0, 1);
    device.run("pcg.rigid.group_l1", |encoder| {
        produce(encoder)?;
        if body_base > 0 {
            // Safety: `reduced` and the scratch outlive the call.
            unsafe { encode_fold(encoder, reduced, body_base, fold, cloth_slot, true) }?;
        }
        if bodies > 0 {
            let body_slot = norms.span(1, bodies);
            let args = PcgRigidGroupL1Args {
                reduced,
                body_base: body_base as u32,
                norm: body_slot,
                count: bodies as u32,
                seam_arena_count: 0,
            };
            // Safety: both handles name live allocations for the whole call.
            unsafe { encoder.elements(&args, bodies as u32) }?;
        }
        Ok(())
    })?;
    norms.download(device)?;
    let host = norms.host();
    out[0] = if body_base > 0 { host[0] } else { 0.0 };
    for body in 0..bodies {
        out[1 + body] = host[1 + body];
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub struct InertialGuessContext { pub eval_x: ppf_cts_compute::Handle, pub target: ppf_cts_compute::Handle, pub prop: ppf_cts_compute::Handle, pub pull: ppf_cts_compute::Handle, pub pull_count: usize }

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FullBodyAdmission { Unqualified, ExistingContact, AnchoredFreeFlight }

pub unsafe fn solve_rigid<D: Device>(
    device: &mut D, op: &Operator, reduction: &mut super::pdrd::Reduction<'_>,
    factor: ppf_cts_compute::Handle, inverse_diagonal: ppf_cts_compute::Handle,
    b: ppf_cts_compute::Handle, x: ppf_cts_compute::Handle,
    work: &mut Workspace, rigid: &mut RigidWorkspace,
    rotation_out: Option<ppf_cts_compute::Handle>,
    lock: Option<&mut super::lock::Projector<'_>>, joints: super::joints::View,
    roots: Option<super::roots::View>,
    locks: ppf_cts_compute::Handle, tolerance: f32, max_iterations: u32,
    arithmetic_eligible: bool, resident_without_hard: bool, dof_mask: ppf_cts_compute::Handle,
    inertial_context: Option<InertialGuessContext>,
    full_body_admission: FullBodyAdmission,
) -> super::scene::FatalResult<Report> {
    solve_rigid_impl(device, op, reduction, factor, inverse_diagonal, b, x,
                     work, rigid, rotation_out, lock, joints, roots, locks,
                     tolerance, max_iterations, RigidSolveInput::NewtonForce, arithmetic_eligible, resident_without_hard, Some(dof_mask), inertial_context, full_body_admission)
}

/// Pure inverse action for a hard-constraint Schur column on the SAME frozen
/// coupled operator as the Newton solve. Both rhs and output are reduced `dim`
/// float vectors including flexible DOFs; the RHS already contains B^T lambda.
/// No force restriction, joint-gradient addition, prescribed displacement seed,
/// affine particular correction, or rotation export belongs to this call.
/// The caller must keep operator, reduction fit, joint view, preconditioner and
/// lock projector frozen across every column and reject unconverged reports.
/// This function does not claim that a truncated solve is an inverse.
///
/// # Safety
/// In addition to solve_rigid's invariants, rhs/output must each span `dim`
/// floats, must not alias the workspace scratch and must outlive dispatches.
#[allow(clippy::too_many_arguments)]
pub unsafe fn solve_rigid_reduced_rhs_with_roots<D: Device>(
    device: &mut D, op: &Operator, reduction: &mut super::pdrd::Reduction<'_>,
    factor: ppf_cts_compute::Handle, inverse_diagonal: ppf_cts_compute::Handle,
    rhs: ppf_cts_compute::Handle, output: ppf_cts_compute::Handle,
    work: &mut Workspace, rigid: &mut RigidWorkspace,
    lock: Option<&mut super::lock::Projector<'_>>, joints: super::joints::View,
    roots: Option<super::roots::View>,
    tolerance: f32, max_iterations: u32,
) -> super::scene::FatalResult<Report> {
    solve_rigid_impl(device, op, reduction, factor, inverse_diagonal, rhs, output,
                     work, rigid, None, lock, joints, roots, ppf_cts_compute::Handle::NONE,
                     tolerance, max_iterations, RigidSolveInput::ReducedRhs, false, false, None, None, FullBodyAdmission::Unqualified)
}

pub unsafe fn solve_rigid_reduced_rhs<D: Device>(
    device: &mut D, op: &Operator, reduction: &mut super::pdrd::Reduction<'_>,
    factor: ppf_cts_compute::Handle, inverse_diagonal: ppf_cts_compute::Handle,
    rhs: ppf_cts_compute::Handle, output: ppf_cts_compute::Handle,
    work: &mut Workspace, rigid: &mut RigidWorkspace,
    lock: Option<&mut super::lock::Projector<'_>>, joints: super::joints::View,
    tolerance: f32, max_iterations: u32,
) -> super::scene::FatalResult<Report> {
    solve_rigid_impl(device, op, reduction, factor, inverse_diagonal, rhs, output,
                     work, rigid, None, lock, joints, None, ppf_cts_compute::Handle::NONE,
                     tolerance, max_iterations, RigidSolveInput::ReducedRhs, false, false, None, None, FullBodyAdmission::Unqualified)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RigidSolveInput { NewtonForce, ReducedRhs }

/// Both exact-seed and iterative exits must export the same reduced iterate.
/// A zero residual is not a zero iterate: prescribed root rows can contain
/// nonzero rotation. Use the affine/canonicalized iterate actually prolonged.
unsafe fn export_rigid_solve_iterate<D: Device>(
    device: &mut D,
    reduction: &mut super::pdrd::Reduction<'_>,
    reduced: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    rotation_out: Option<ppf_cts_compute::Handle>,
    input: RigidSolveInput,
) -> Result<(), Fault> {
    if input == RigidSolveInput::NewtonForce {
        reduction.prolong(device, reduced, x)?;
        if let Some(rotation) = rotation_out {
            reduction.extract_body_rotation(device, reduced, rotation)?;
        }
    } else {
        combine(device, reduced, reduced, x, 1.0, 0.0, reduction.map.dim)?;
    }
    Ok(())
}

fn fem_full_space_identity(map:&super::rigid_map::RigidMap)->bool {
    map.nrow>0 && map.n_bodies==0 && map.n_cloth==map.nrow &&
    map.body_base==3*map.nrow && map.dim==3*map.nrow &&
    !map.any_joint && !map.any_translation_lock && !map.any_rotation_lock &&
    map.vbody.len()==map.nrow && map.cloth_off.len()==map.nrow &&
    map.vbody.iter().all(|v|*v==super::rigid_map::RIGID_UNSET) &&
    map.cloth_off.iter().enumerate().all(|(i,o)|*o as usize==3*i) &&
    map.jmode.is_empty() && map.jaxis.is_empty() &&
    map.tlock.is_empty() && map.tlock_axis.is_empty() && map.tlock_mode.is_empty() &&
    map.rlock.is_empty() && map.rlock_axis.is_empty() && map.rlock_mode.is_empty()
}

#[path="true_residual_capture.rs"]
mod true_residual_capture;
#[path="shadow_residual_diagnostics.rs"]
mod shadow_residual_diagnostics;

unsafe fn solve_rigid_impl<D: Device>(
    device: &mut D,
    op: &Operator,
    reduction: &mut super::pdrd::Reduction<'_>,
    factor: ppf_cts_compute::Handle,
    inverse_diagonal: ppf_cts_compute::Handle,
    b: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    work: &mut Workspace,
    rigid: &mut RigidWorkspace,
    rotation_out: Option<ppf_cts_compute::Handle>,
    // THE PROJECTED LOCK, when the scene carries one.
    //
    // A LOCKED REDUCED SOLVE IS THE SAME LOOP, not a second algorithm. What
    // differs is that the reduced operator projects the FULL-SPACE product
    // before restricting it, and that the seed carries the lock's particular
    // solution. Composing the two projectors in the other order would not be the
    // same operator, which is why the lock's projection sits inside the reduced
    // apply rather than around it.
    lock: Option<&mut super::lock::Projector<'_>>,
    joints: super::joints::View,
    roots: Option<super::roots::View>,
    // The scene's `TranslationLock` records, which the particular term reads
    // for each locked body's total mass.
    locks: ppf_cts_compute::Handle,
    tolerance: f32,
    max_iterations: u32,
    input: RigidSolveInput,
    arithmetic_eligible: bool, resident_without_hard: bool, warm_mask: Option<ppf_cts_compute::Handle>,
    inertial_context: Option<InertialGuessContext>,
    full_body_admission: FullBodyAdmission,
) -> super::scene::FatalResult<Report> {
    rigid.body_host_tree=None;
    rigid.active_factor_key=None;
    rigid.active_factor_borrowed=false;
    rigid.active_factor_prepare_ms=0.0;
    rigid.active_factor_iteration_debt = None;
    rigid.active_factor_fresh_prefix = None;
    let old=if let Some(mut owner)=rigid.body_inverse.take(){owner.free(device)}else{Ok(())};
    let old_factor=if let Some(mut owner)=rigid.body_factor.take(){owner.free(device)}else{Ok(())};
    let old_x=rigid.body_seed_x.free(device);let old_r=rigid.body_seed_r.free(device);
    old?;old_factor?;old_x?;old_r?;
    let result=solve_rigid_inner(device,op,reduction,factor,inverse_diagonal,b,x,work,rigid,rotation_out,lock,joints,roots,locks,tolerance,max_iterations,input,arithmetic_eligible,resident_without_hard,warm_mask,inertial_context,full_body_admission);
    rigid.body_inverse_debug=None;rigid.body_group_reference=None;
    if let Some(host)=rigid.body_host_tree.take(){host.receipt();}
    let mut cleanup_error = None;
    macro_rules! collect_cleanup {
        ($operation:expr) => {
            if let Err(error) = $operation {
                if cleanup_error.is_none() { cleanup_error = Some(error); }
            }
        };
    }
    if let Some(mut owner) = rigid.body_inverse.take() { collect_cleanup!(owner.free(device)); }
    let success=matches!(&result,Ok(Report{outcome:Outcome::Converged | Outcome::AcceptedAnimation,..}));
    let mut pending = rigid.body_factor.take();
    let publish = success && pending.is_some() && rigid.active_factor_key.is_some();
    if publish {
        if let Some(mut old) = rigid.body_factor_cache.take() { collect_cleanup!(old.owner.free(device)); }
    } else if !success || rigid.active_factor_key.is_none() && rigid.body_factor_cache.is_some() && rigid.active_factor_prepare_ms==0.0 {
        collect_cleanup!(rigid.release_factor_cache(device));
    }
    collect_cleanup!(rigid.body_seed_x.free(device));
    collect_cleanup!(rigid.body_seed_r.free(device));
    if !publish || cleanup_error.is_some() || result.is_err() {
        if let Some(mut owner) = pending.take() { collect_cleanup!(owner.free(device)); }
    }
    if publish && cleanup_error.is_none() && result.is_ok() {
        let iteration_debt = match &result {
            Ok(Report { outcome: Outcome::Converged, iterations, .. }) => {
                match rigid.active_factor_fresh_prefix {
                    Some(prefix) => Some(FactorIterationDebt {
                        dt_bits: joints.dt.to_bits(),
                        fresh_iterations: iterations.saturating_sub(prefix).max(1),
                        excess_iterations: 0,
                    }),
                    None => rigid.active_factor_iteration_debt.map(|debt| debt.charge(*iterations)),
                }
            }
            _ => None,
        };
        rigid.body_factor_cache=Some(CachedBodyFactor{owner:pending.take().unwrap(),key:rigid.active_factor_key.take().unwrap(),prepare_ms:rigid.active_factor_prepare_ms,iteration_debt});
    }
    match result { Err(error)=>Err(error), Ok(value)=>{if let Some(error)=cleanup_error{return Err(error.into());} Ok(value)} }

}

unsafe fn solve_rigid_inner<D: Device>(
    device: &mut D,
    op: &Operator,
    reduction: &mut super::pdrd::Reduction<'_>,
    factor: ppf_cts_compute::Handle,
    inverse_diagonal: ppf_cts_compute::Handle,
    b: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    work: &mut Workspace,
    rigid: &mut RigidWorkspace,
    rotation_out: Option<ppf_cts_compute::Handle>,
    // THE PROJECTED LOCK, when the scene carries one.
    //
    // A LOCKED REDUCED SOLVE IS THE SAME LOOP, not a second algorithm. What
    // differs is that the reduced operator projects the FULL-SPACE product
    // before restricting it, and that the seed carries the lock's particular
    // solution. Composing the two projectors in the other order would not be the
    // same operator, which is why the lock's projection sits inside the reduced
    // apply rather than around it.
    lock: Option<&mut super::lock::Projector<'_>>,
    joints: super::joints::View,
    roots: Option<super::roots::View>,
    // The scene's `TranslationLock` records, which the particular term reads
    // for each locked body's total mass.
    locks: ppf_cts_compute::Handle,
    tolerance: f32,
    max_iterations: u32,
    input: RigidSolveInput,
    arithmetic_eligible: bool, resident_without_hard: bool, warm_mask: Option<ppf_cts_compute::Handle>,
    inertial_context: Option<InertialGuessContext>,
    full_body_admission: FullBodyAdmission,
) -> super::scene::FatalResult<Report> {
    let lanczos_solve_id = lanczos_trace::solve_id();
    let dim = reduction.map.dim;
    let bodies = reduction.map.n_bodies;
    let body_base = reduction.map.body_base;

    // Stage1 sync coalescing only: do not alter the old scalar recurrence,
    // zero-prp fallback, seeded L1, cap or any positive-PDRD/constraint path.
    let mut grouped_soft = joints.soft.filter(|soft|
        input==RigidSolveInput::NewtonForce && lock.is_none() && roots.is_none() &&
        joints.count==0 && joints.control_count==0 && joints.extension_count==0 &&
        fem_full_space_identity(reduction.map) &&
        soft.ready && soft.bodies==0 && soft.vertices as usize==op.rows() as usize &&
        soft.vertices==soft.free_count && soft.dim as usize==dim && soft.base as usize==dim);

    // NewtonForce mixed PDRD path: one readback carries both
    // group L1 and the next r.z. ReducedRhs (including hard Schur), projected
    // bodies, roots, locks, and Schwarz systems retain the scalar path. Joints
    // and restoring terms still participate through the ordinary operator.
    // The extra preconditioner application is speculative: its scratch result
    // is ignored if this iteration converges or stops at the cap.
    // Default to the measured CUDA path. Keep explicit 0 and 1 overrides for
    // reproducible diagnostics and preserve the previous opt-in on other backends.
    let coalesce_enabled = match std::env::var_os("PPF_PCG_COALESCE_MIXED").as_deref() {
        Some(value) if value == std::ffi::OsStr::new("1") => true,
        Some(_) => false,
        None => cfg!(cuda_backend),
    };
    let mut coalesce_mixed = coalesce_enabled
        && input == RigidSolveInput::NewtonForce
        && bodies > 0 && !reduction.map.needs_projection()
        && lock.is_none() && roots.is_none()
        && !reduction.staged.reduced_schwarz.active();
    let fused_joint_product = std::env::var_os("PPF_JOINT_FUSED_PRODUCT")
        .is_some_and(|value| value == "1")
        && (joints.count != 0 || joints.soft.is_some())
        && joints.can_encode_krylov()
        && lock.is_none() && roots.is_none() && !reduction.map.needs_projection();
    // Optional search-product optimizations share these safety gates. The
    // original affine and true-residual operators below remain unchanged.
    let safe_krylov = input == RigidSolveInput::NewtonForce && resident_without_hard
        && reduction.staged.pcg_precision == crate::args::PcgPrecision::Fp32
        && lock.is_none() && !reduction.map.needs_projection()
        && std::env::var_os("PPF_RIGID_PCG_FAILURE_CAPTURE_DIR").is_none()
        && std::env::var_os("PPF_PCG_TRUE_RESIDUAL_CAPTURE_DIR").is_none()
        && std::env::var_os("PPF_PCG_TRUE_RESIDUAL_DIAGNOSTICS").is_none()
        && std::env::var_os("PPF_PCG_TRUE_RESIDUAL_CAPTURE_STOP").is_none();
    let safe_hot = safe_krylov && roots.is_none();
    let host_tet_operator = device.info().backend == "host" && op.has_tet_factor();
    let cpu_force_only = safe_krylov && host_tet_operator
        && std::env::var_os("PPF_CPU_KRYLOV_TET_FORCE_ONLY").is_some_and(|value| value == "1");
    let mut cpu_force_only_receipt = cpu_force_only
        && std::env::var_os("PPF_TET_FORCE_ONLY_RECEIPT").is_some_and(|value| value == "1");
    let timing_receipt = if std::env::var_os("PPF_PCG_TIMING_RECEIPT").as_deref()
        == Some(std::ffi::OsStr::new("1")) {
        Some((std::time::Instant::now(), device.counters()))
    } else { None };
    // Taking the key invalidates the previous solve immediately: failure,
    // non-Newton calls and all non-converged exits cannot leave stale history.
    let requested_mode = match std::env::var_os("PPF_PCG_WARM_START").as_deref() {
        Some(v) if v == "1" => 1,
        Some(v) if v == "2" => 2,
        _ => 0,
    };
    let warm_enabled = requested_mode != 0;
    let warm_auto_cost = requested_mode == 2 &&
        std::env::var_os("PPF_PCG_WARM_AUTO_COST").is_some_and(|v| v == "1");
    let warm_requested = warm_enabled && safe_hot && warm_mask.is_some() && joints.dt.is_finite() && joints.dt > 0.0;
    if warm_enabled && input == RigidSolveInput::NewtonForce && !warm_requested {
        let reason = if !resident_without_hard {"hard_or_capture"}
            else if reduction.staged.pcg_precision != crate::args::PcgPrecision::Fp32 {"precision"}
            else if lock.is_some() {"lock"} else if roots.is_some() {"root"}
            else if reduction.map.needs_projection() {"body_projection"}
            else if warm_mask.is_none() {"mask_unavailable"}
            else if !joints.dt.is_finite() || joints.dt <= 0.0 {"dt_invalid"}
            else {"diagnostic"};
        eprintln!("PPF_PCG_WARM_START disabled reason={}", reason);
    }
    // Invalidate before allocation; a Device allocation fault must abort.
    let prior_cold_cost = std::mem::take(&mut rigid.warm_cold_cost);
    let (prior_warm_key,prior_warm_count) = rigid.begin_warm(device,dim,if warm_requested {requested_mode}else{0})?;
    let mut warm_key = None;
    if warm_requested {
        let allocated = true;
        if allocated {
            let mut mask = Vec::<u32>::new();
            if mask.try_reserve_exact(reduction.map.nrow).is_ok() {
                mask.resize(reduction.map.nrow, 0);
                if !mask.is_empty() {
                    let bytes = std::slice::from_raw_parts_mut(mask.as_mut_ptr().cast::<u8>(), mask.len()*4);
                    device.read(warm_mask.unwrap(), 0, bytes)?;
                }
                if let Some(free_rows) = warm_start_policy::reduced_free_rows(reduction.map, &mask) {
                    let m = reduction.map;
                    let (mut vbody,mut cloth_off)=(Vec::new(),Vec::new());
                    if vbody.try_reserve_exact(m.vbody.len()).is_ok() && cloth_off.try_reserve_exact(m.cloth_off.len()).is_ok() {
                        vbody.extend_from_slice(&m.vbody); cloth_off.extend_from_slice(&m.cloth_off);
                        rigid.warm_free_rows.write(device, 0, &free_rows)?;
                        warm_key = Some(WarmKey { nrow:m.nrow, n_cloth:m.n_cloth, bodies:m.n_bodies,
                            dim:m.dim, body_base:m.body_base, vbody, cloth_off, mask, dt_bits:joints.dt.to_bits(),mode:requested_mode });
                    } else { eprintln!("PPF_PCG_WARM_START disabled reason=host_allocation"); }
                } else { eprintln!("PPF_PCG_WARM_START disabled reason=mask_map_invalid"); }
            } else { eprintln!("PPF_PCG_WARM_START disabled reason=host_allocation"); }
        }
    }
    let warm_eligible = warm_key.is_some();
    let warm_has_history = warm_eligible && prior_warm_count > 0 && prior_warm_key.as_ref() == warm_key.as_ref();
    let usable_warm_count = if warm_has_history {prior_warm_count} else {0};
    let mut cold_cost = ColdCost::for_key(warm_auto_cost,prior_warm_key.as_ref(),warm_key.as_ref(),prior_cold_cost);
    let auto_candidate = !warm_auto_cost || cold_cost.allow_candidate();
    if warm_auto_cost && warm_eligible {
        let (count,recent)=cold_cost.samples();
        eprintln!("PPF_PCG_WARM_AUTO_COST decision={} cold_samples={} recent={:?}",
            if warm_has_history && auto_candidate {"attempt"} else {"cold"},count,recent);
    }
    let tet_force_only = safe_hot && op.tet_factor.is_some()
        && std::env::var_os("PPF_TET_FORCE_ONLY").is_some_and(|v| v == "1");
    let row_parallel = safe_hot && std::env::var_os("PPF_JOINT_ROW_PARALLEL").is_some_and(|v| v == "1");
    let mut row_receipt_pending = row_parallel && joints.soft.is_some_and(|soft| soft.row_parallel_available())
        && std::env::var_os("PPF_JOINT_ROW_PARALLEL_RECEIPT").is_some_and(|v| v == "1");
    let mut tet_receipt_pending = tet_force_only
        && std::env::var_os("PPF_TET_FORCE_ONLY_RECEIPT").is_some_and(|v| v == "1");
    let allow_gather = safe_hot && reduction.staged.restrict_gather_enabled;
    let mut gather_receipt_pending = allow_gather
        && std::env::var_os("PPF_RESTRICT_GATHER_RECEIPT").is_some_and(|v| v == "1");
    let mut coalesced_iterations = 0u32;
    // Reserve iterations for an exact-operator FP32 restart.
    let arithmetic_limit = max_iterations.saturating_sub((max_iterations / 4).max(1));
    let arithmetic_eligible = arithmetic_eligible && arithmetic_limit > 0;
    let mut arithmetic_active = arithmetic_eligible;
    let observe_arithmetic = reduction.staged.pcg_precision == crate::args::PcgPrecision::MixedFp16Arithmetic
        && std::env::var_os("PPF_PCG_ARITHMETIC_RECEIPT").is_some_and(|value| value == "1");
    let record_arithmetic = arithmetic_eligible && observe_arithmetic;
    let mut arithmetic_downgrade = None;
    let mut arithmetic_applies = 0u32;
    let mut lock = lock;
    // Scope intentionally matches the sealed failing two-body/KINroot fixture.
    // Other composition paths need their own diagnostic projector contract.
    let failure_capture_dir = if input == RigidSolveInput::NewtonForce
        && dim == 12 && bodies == 2 && body_base == 0 && lock.is_none() && roots.is_some()
    {
        std::env::var("PPF_RIGID_PCG_FAILURE_CAPTURE_DIR").ok().filter(|v| !v.is_empty())
    } else { None };
    let failure_handles = RigidFailureHandles {
        r: rigid.r.handle(), z: rigid.z.handle(), p: rigid.p.handle(),
        ap: rigid.rp.handle(), rhs: rigid.f.handle(), x: rigid.xr.handle(),
        blocks: rigid.failure_precond_blocks,
    };
    let mut failure_first = serde_json::Value::Null;
    let mut failure_cap_before = serde_json::Value::Null;
    let mut failure_previous_beta: Option<f32> = None;

    let auto_tree = runtime_auto_policy::candidate(runtime_auto_policy::Domain {
        cuda: cfg!(cuda_backend), fp32: reduction.staged.pcg_precision == crate::args::PcgPrecision::Fp32,
        newton: input == RigidSolveInput::NewtonForce, bodies: bodies > 0,
        joints: joints.count != 0 && joints.can_encode_krylov(),
        compatible: resident_without_hard && lock.is_none() && !reduction.map.needs_projection()
            && !reduction.staged.reduced_schwarz.active(),
        explicit: runtime_auto_policy::explicit_controls(),
    });
    let iteration_debt_policy = auto_tree
        && std::env::var_os("PPF_BODY_TREE_REFRESH_POLICY").is_some_and(|value| value == "iteration-debt");

    // THE TANGENT PROJECTION. It is the projector every Krylov vector in a
    // LOCKED solve passes through, and it is not the same thing as
    // `project_bodies`.
    //
    // THE TWO HALVES LIVE IN DIFFERENT SPACES, which is the whole reason it has
    // five steps. A body's constraint rows are expressed in the REDUCED basis,
    // so `project_bodies` handles them; the aggregate lock's rows are built over
    // VERTICES, so they can only be applied in the full space. The vector is
    // therefore projected in the reduced basis, prolonged, projected again in
    // the full one, and only its CLOTH rows brought back, because the body rows
    // were already correct and the prolonged image would overwrite them.
    //
    // WITHOUT A LOCK IT DEGENERATES TO `project_bodies`, which is why the
    // unlocked arm is untouched by this: the middle three steps would prolong,
    // apply nothing, and copy the cloth rows back unchanged.
    macro_rules! project_tangent {
        ($vector:expr) => {{
            reduction.project_bodies(device, $vector)?;
            if let Some(root)=roots { root.project(device, $vector)?; }
            if let Some(projector) = lock.as_deref_mut() {
                reduction.prolong(device, $vector, rigid.xv.handle())?;
                projector.project(device, rigid.xv.handle(), 1)?;
                reduction.copy_projected_cloth(device, rigid.xv.handle(), $vector)?;
                reduction.project_bodies(device, $vector)?;
                if let Some(root)=roots { root.project(device, $vector)?; }
            }
        }};
    }

    // Affine seed and residual verification retain the original operator and
    // its inhomogeneous components. Only the Krylov product uses the cache.
    macro_rules! apply_reduced {
        ($input:expr, $output:expr, $krylov:literal) => {{
            apply_reduced!($input, $output, $krylov, false);
        }};
        ($input:expr, $output:expr, $krylov:literal, $force_only:expr) => {{
            if auto_tree && !$krylov {
                // Retain the full native operator, including the tet factor.
                // Only eliminate command boundaries between its ordered stages.
                device.run("pcg.rigid.reference_native", |encoder| {
                    reduction.encode_prolong(encoder, $input, rigid.xv.handle())?;
                    op.encode_apply(encoder, rigid.xv.handle(), rigid.mxv.handle(),
                        work.row_absolute.handle(), work.row_curvature.handle())?;
                    reduction.encode_restrict(encoder, rigid.mxv.handle(), $output, true)
                })?;
            } else {
                reduction.prolong(device, $input, rigid.xv.handle())?;
                {
                    let _phase = if host_tet_operator {
                        super::phase::start(if $krylov {
                            "pcg.cpu.krylov_native_product"
                        } else {
                            "pcg.cpu.affine_native_product"
                        })
                    } else {
                        None
                    };
                    if $krylov && $force_only {
                        // The ordinary reduced recurrence computes p.Ap after
                        // restriction, joints and Q. It never uses the tet
                        // gather's per-row curvature or magnitude arrays.
                        device.run("operator.apply", |encoder| {
                            op.encode_apply_force_only(encoder, rigid.xv.handle(), rigid.mxv.handle(),
                                work.row_absolute.handle(), work.row_curvature.handle())
                        })?;
                        if cpu_force_only_receipt {
                            eprintln!("CPU_TET_FORCE_ONLY_RECEIPT active=true roots={}", roots.is_some());
                            cpu_force_only_receipt = false;
                        }
                    } else {
                        op.apply(device, rigid.xv.handle(), rigid.mxv.handle(),
                                 work.row_absolute.handle(), work.row_curvature.handle())?;
                    }
                }
                // The lock projects the product before restriction because its
                // rows are expressed in the full vertex space.
                if let Some(projector) = lock.as_deref_mut() {
                    projector.project(device, rigid.mxv.handle(), 1)?;
                }
                reduction.restrict(device, rigid.mxv.handle(), $output, &rigid.zero)?;
            }
            if $krylov {
                joints.add_krylov(device, $input, $output)?;
            } else {
                joints.add(device, 1, $input, $output)?;
            }
            reduction.project_bodies(device, $output)?;
            if let Some(root)=roots { root.project(device, $output)?; }
        }};
    }

    macro_rules! measure_reference_residual {
        ($group:ident) => {{
            let rhs = rigid.f.handle();
            let product = rigid.rp.handle();
            let residual = rigid.r.handle();
            if auto_tree {
                group_l1_after(device, residual, &mut rigid.group_norm, &work.fold,
                    body_base, bodies, &mut rigid.$group, |encoder| {
                        encode_combine_raw(encoder, rhs, product, residual, 1.0, -1.0, dim)?;
                        if let Some(root) = roots { root.encode_project(encoder, residual)?; }
                        Ok(())
                    })?;
            } else {
                combine(device, rhs, product, residual, 1.0, -1.0, dim)?;
                project_tangent!(residual);
                group_l1_device(device, residual, &mut rigid.group_norm,
                    &work.fold, body_base, bodies, &mut rigid.$group)?;
            }
        }};
    }

    // `R v` WITH its input projected first, and the only operator the CG loop
    // may use.
    //
    // PROJECTING BOTH SIDES IS WHAT MAKES IT SYMMETRIC. The loop's operator is
    // `Q M Q`; projecting only the output gives `Q M`, which is NOT symmetric
    // whenever `Q` is not the identity, and conjugate gradients on a
    // non-symmetric operator has no descent guarantee and no reason to converge.
    // Without a lock `project_tangent` is `project_bodies`, every Krylov vector
    // already carries it, and the two applies agree.
    macro_rules! apply_reduced_tangent {
        ($input:expr, $output:expr) => {{
            project_tangent!($input);
            apply_reduced!($input, $output, true, cpu_force_only);
        }};
    }

    // The Newton arm preserves its existing force assembly and seed. A
    // Schur inverse column is already reduced and must remain a linear map.
    if input == RigidSolveInput::NewtonForce {
        reduction.restrict(device, b, rigid.f.handle(), &rigid.zero)?;
        joints.add(device, 0, rigid.f.handle(), rigid.f.handle())?;
        reduction.seed_restrict(device, x, rigid.xr.handle(), &rigid.zero)?;
    } else {
        combine(device, b, b, rigid.f.handle(), 1.0, 0.0, dim)?;
        rigid.xr.write(device, 0, &rigid.zero[..dim])?;
    }
    reduction.project_bodies(device, rigid.f.handle())?;
    if let Some(root)=roots { root.project(device, rigid.f.handle())?; }
    // THE LOCK'S PARTICULAR SOLUTION, BEFORE THE BODY PROJECTION. A locked
    // body's three translation DOFs are not free: the lock's drift over its
    // total mass IS their value, so the seed carries it and the CG solves only
    // the homogeneous part. It is written between the seed's restriction and the
    // first residual, so the residual is measured against a seed that already
    // satisfies the lock.
    if input == RigidSolveInput::NewtonForce {
        if let Some(projector) = lock.as_deref_mut() {
            reduction.translation_lock_particular(
                device, locks, projector.scratch.drift.handle(), rigid.xr.handle(),
            )?;
        }
    }
    reduction.project_bodies(device, rigid.xr.handle())?;
    if let Some(root)=roots { root.project(device, rigid.xr.handle())?; }
    if input == RigidSolveInput::NewtonForce {
        if let Some(root)=roots { root.particular(device, joints.state, rigid.xr.handle())?; }
    }
    // THE AFFINE APPLY, and the one place it is used. The seed carries the
    // lock's inhomogeneous particular correction, which the tangent projector is
    // built to remove, so projecting the input here would discard exactly the
    // part the residual has to see.
    apply_reduced!(rigid.xr.handle(), rigid.rp.handle(), false);
    // The seed residual is projected before `err0` is taken from it, for the
    // reason the per-iteration projection below states.
    measure_reference_residual!(group_initial);
    let err0: f32 = rigid.group_initial.iter().sum();
    if err0 == 0.0 {
        rigid.full_body_reuse = full_body_native::Reuse::default();
        if warm_requested { eprintln!("PPF_PCG_WARM_START cold reason=zero_residual"); }
        // Already exact. The seed is prolonged back out rather than `x` being
        // zeroed, which would discard the prescribed rows the caller loaded.
        if std::env::var_os("PPF_LEGACY_ROWS_SELF_CHECK").as_deref()==Some(std::ffi::OsStr::new("1")) {return Err(super::scene::Fatal::invariant("legacy rows self check not reached; no commit"));}
        if std::env::var_os("PPF_FACTOR_BATCH8_PACKET_SELF_CHECK").as_deref()==Some(std::ffi::OsStr::new("1")) {return Err(super::scene::Fatal::invariant("factor batch8 diagnostic not reached; no commit"));}
        export_rigid_solve_iterate(device, reduction, rigid.xr.handle(), x, rotation_out, input)?;
        if let Some((started, before)) = timing_receipt {
            let after = device.counters();
            eprintln!("PPF_PCG_TIMING_RECEIPT input={} coalesce_eligible={} coalesced_iterations=0 bodies={} body_projection={} roots={} lock={} schwarz={} outcome=Converged iterations=1 true_residual_checks=0 residual=0 elapsed_ms={:.3} syncs={} dispatches={} readback_requests=unavailable readback_bytes=unavailable",
                if input == RigidSolveInput::NewtonForce {"newton"} else {"reduced_rhs"},
                coalesce_mixed,bodies,reduction.map.needs_projection(),roots.is_some(),lock.is_some(),
                reduction.staged.reduced_schwarz.active(),started.elapsed().as_secs_f64()*1000.0,
                after.syncs.saturating_sub(before.syncs),after.dispatches.saturating_sub(before.dispatches));
        }
        return Ok(Report {
            outcome: Outcome::Converged,
            iterations: 1,
            relative_residual: 0.0,
        });
    }

    if let Some(selected) = std::env::var("PPF_BODY_FULL_FACTOR_PROBE")
        .ok().and_then(|value| value.parse::<usize>().ok())
    {
        static SOLVES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let solve = SOLVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if solve == selected {
            if input != RigidSolveInput::NewtonForce || !resident_without_hard
                || lock.is_some() || reduction.map.needs_projection()
                || reduction.staged.reduced_schwarz.active()
                || reduction.staged.pcg_precision != crate::args::PcgPrecision::Fp32
                || body_base != 0 || bodies == 0 || bodies.checked_mul(6) != Some(dim)
                || op.has_tet_factor() || joints.soft.is_some()
            {
                return Err(super::scene::Fatal::invariant("full-body factor probe unsupported; no commit"));
            }
            let before = super::shadow_tree::fault_snapshot(device, op, joints)
                .map_err(super::scene::Fatal::invariant)?;
            if before.iter().flatten().any(|&fault| fault != 0) {
                return Err(super::scene::Fatal::invariant("full-body factor probe found native fault; no commit"));
            }
            let mut fixed = vec![false; bodies];
            if let Some(root) = roots {
                let mut targets = vec![crate::joint_root_motion_gpu::RootTargetGpu::default(); root.count as usize];
                let byte_count = std::mem::size_of_val(targets.as_slice());
                let bytes = std::slice::from_raw_parts_mut(targets.as_mut_ptr().cast::<u8>(), byte_count);
                if !bytes.is_empty() { device.read(root.targets, 0, bytes)?; }
                for t in targets.into_iter().filter(|t| t.enabled != 0) {
                    *fixed.get_mut(t.body as usize)
                        .ok_or_else(|| super::scene::Fatal::invariant("root outside body layout"))? = true;
                }
            }
            let free: Vec<_> = (0..bodies).filter(|&b| !fixed[b]).collect();
            let slots: Vec<_> = free.iter().flat_map(|&body| (0..6).map(move |lane| 6*body+lane)).collect();
            let started = std::time::Instant::now();
            let plan = super::body_block_capture::host_sparse_plan(op, joints, bodies, free.len(), roots.map_or(0, |root| root.count as usize)).ok_or_else(|| super::scene::Fatal::invariant("full factor capture budget; no commit"))?; let matrix = super::body_block_capture::capture_sparse(device, op, reduction, joints, &slots, Some(plan))?;
            let capture_ms = 1000.0 * started.elapsed().as_secs_f64();
            let x0 = rigid_failure_read_f32(device, rigid.xr.handle(), dim)?;
            let r0 = rigid_failure_read_f32(device, rigid.r.handle(), dim)?;
            if let Some(destination) = std::env::var_os("PPF_BODY_FULL_FACTOR_MATRIX") {
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new().write(true).create_new(true)
                    .open(destination).map_err(|error| super::scene::Fatal::invariant(error.to_string()))?;
                let blocks: Vec<_> = matrix.blocks.iter().map(|(&(a,b), block)| (a,b,block)).collect();
                writeln!(file, "{}", serde_json::json!({
                    "solve": solve, "bodies": bodies, "free": free, "blocks": blocks,
                    "x0": x0, "r0": r0, "initial_groups": rigid.group_initial,
                    "err0": err0, "tolerance": tolerance, "capture_ms": capture_ms,
                    "original_operator": "native_fp32_projected_affine",
                })).map_err(|error| super::scene::Fatal::invariant(error.to_string()))?;
            }
            let started = std::time::Instant::now();
            let full = full_body_factor::Factor::build(free.len(), &matrix.blocks,
                full_body_factor::Budget { bytes: 64 << 20, fill_blocks: 16384 });
            let full_ms = 1000.0 * started.elapsed().as_secs_f64();
            eprintln!("PPF_BODY_FULL_FACTOR_CAPTURE solve={solve} bodies={bodies} free={} directed={} capture_ms={capture_ms:.6}",
                free.len(), matrix.blocks.len());
            match &full {
                Ok(factor) => eprintln!("PPF_BODY_FULL_FACTOR_SETUP mode=full factor_ms={full_ms:.6} stats={:?}", factor.stats),
                Err(reason) => eprintln!("PPF_BODY_FULL_FACTOR_REFUSED mode=full factor_ms={full_ms:.6} reason={reason}"),
            }
            let started = std::time::Instant::now();
            let forest = super::shadow_tree::FreeForest::from_sparse_weighted(&matrix, free.len());
            let forest_ms = 1000.0 * started.elapsed().as_secs_f64();
            match &forest {
                Ok(factor) => eprintln!("PPF_BODY_FULL_FACTOR_SETUP mode=forest factor_ms={forest_ms:.6} omitted={}", factor.omitted_cycles),
                Err(reason) => eprintln!("PPF_BODY_FULL_FACTOR_REFUSED mode=forest factor_ms={forest_ms:.6} reason={:?}", reason.0),
            }
            let mut trial = Buffer::<f32>::none();
            let mut product = Buffer::<f32>::none();
            let mut residual = Buffer::<f32>::none();
            let attempted = (|| -> super::scene::FatalResult<()> {
                for buffer in [&mut trial, &mut product, &mut residual] {
                    buffer.size(device, dim, ppf_cts_compute::AllocLabel("body.full_factor_probe"))?;
                }
                for mode in ["full", "forest"] {
                    if (mode == "full" && full.is_err()) || (mode == "forest" && forest.is_err()) { continue; }
                    let mut x = x0.clone();
                    let mut r = r0.clone();
                    for pass in 1..=3 {
                        let started = std::time::Instant::now();
                        let correction = if mode == "full" {
                            let rhs: Vec<_> = free.iter().map(|&b| std::array::from_fn(|i| f64::from(r[6*b+i]))).collect();
                            full.as_ref().unwrap().solve(&rhs).map(|delta| {
                                let mut result = vec![0.0f64; dim];
                                for (&body, block) in free.iter().zip(delta) {
                                    result[6*body..6*body+6].copy_from_slice(&block);
                                }
                                result
                            }).map_err(str::to_owned)
                        } else {
                            let rhs: Vec<_> = r.iter().map(|&value| f64::from(value)).collect();
                            forest.as_ref().ok().unwrap().apply(&rhs).map_err(|error| error.0)
                        };
                        let solve_ms = 1000.0 * started.elapsed().as_secs_f64();
                        let delta = match correction {
                            Ok(delta) => delta,
                            Err(reason) => {
                                eprintln!("PPF_BODY_FULL_FACTOR_REFUSED mode={mode} pass={pass} reason={reason}");
                                break;
                            }
                        };
                        for (value, correction) in x.iter_mut().zip(delta) {
                            *value = (f64::from(*value) + correction) as f32;
                        }
                        if x.iter().any(|value| !value.is_finite()) {
                            eprintln!("PPF_BODY_FULL_FACTOR_REFUSED mode={mode} pass={pass} reason=nonfinite_trial");
                            break;
                        }
                        let started = std::time::Instant::now();
                        trial.write(device, 0, &x)?;
                        project_tangent!(trial.handle());
                        if let Some(root) = roots { root.particular(device, joints.state, trial.handle())?; }
                        apply_reduced!(trial.handle(), product.handle(), false);
                        combine(device, rigid.f.handle(), product.handle(), residual.handle(), 1.0, -1.0, dim)?;
                        project_tangent!(residual.handle());
                        group_l1_device(device, residual.handle(), &mut rigid.group_norm, &work.fold,
                            0, bodies, &mut rigid.group_current)?;
                        let score = super::pdrd::worst_relative_residual(&rigid.group_current, &rigid.group_initial, err0);
                        r = rigid_failure_read_f32(device, residual.handle(), dim)?;
                        x = rigid_failure_read_f32(device, trial.handle(), dim)?;
                        let check_ms = 1000.0 * started.elapsed().as_secs_f64();
                        let accepted = score < tolerance;
                        eprintln!("PPF_BODY_FULL_FACTOR_CHECK mode={mode} pass={pass} solve_ms={solve_ms:.6} check_ms={check_ms:.6} true_relative={score:.9e} tolerance={tolerance:.9e} passes={accepted}");
                        if accepted || !score.is_finite() { break; }
                    }
                }
                Ok(())
            })();
            let free_trial = trial.free(device);
            let free_product = product.free(device);
            let free_residual = residual.free(device);
            attempted?;
            free_trial?;
            free_product?;
            free_residual?;
            let after = super::shadow_tree::fault_snapshot(device, op, joints)
                .map_err(super::scene::Fatal::invariant)?;
            if before != after {
                return Err(super::scene::Fatal::invariant("full-body factor probe changed native faults; no commit"));
            }
            return Err(super::scene::Fatal::invariant("full-body factor probe completed; intentional no commit"));
        }
    }


    if std::env::var_os("PPF_BODY_FULL_NATIVE").is_some_and(|value| value == "1")
        && !runtime_auto_policy::explicit_controls()
        && std::env::var_os("PPF_LEGACY_ROWS_SELF_CHECK").is_none()
        && std::env::var_os("PPF_FACTOR_BATCH8_PACKET_SELF_CHECK").is_none()
        && std::env::var_os("PPF_BODY_FACTOR_PACKET_SELF_CHECK").is_none()
        && failure_capture_dir.is_none()
    {
        let trace = std::env::var_os("PPF_BODY_FULL_NATIVE_TRACE").is_some_and(|value| value == "1");
        let contact_only = std::env::var_os("PPF_BODY_FULL_NATIVE_CONTACT_ONLY")
            .is_some_and(|value| value == "1");
        let compatible = input == RigidSolveInput::NewtonForce && resident_without_hard
            && lock.is_none() && !reduction.map.needs_projection()
            && !reduction.staged.reduced_schwarz.active()
            && reduction.staged.pcg_precision == crate::args::PcgPrecision::Fp32
            && body_base == 0 && bodies != 0 && bodies.checked_mul(6) == Some(dim)
            && !op.has_tet_factor() && max_iterations > 0 && !warm_enabled
            && (joints.count != 0 || joints.soft.is_some_and(|soft| soft.count != 0));
        if contact_only && full_body_admission == FullBodyAdmission::Unqualified {
            if trace {
                eprintln!("PPF_BODY_FULL_NATIVE active=false reason=no_dynamic_contact");
            }
        } else if compatible {
            if trace && full_body_admission == FullBodyAdmission::AnchoredFreeFlight {
                eprintln!("PPF_BODY_FULL_ANCHORED_FREE_FLIGHT trial=true");
            }
            let _phase = super::phase::start("pcg.full_body.trial");
            let audit = std::env::var_os("PPF_BODY_FULL_NATIVE_AUDIT_REJECT")
                .is_some_and(|value| value == "1");
            // Read only in the rejection test. "p" also has to survive unchanged
            // for later preconditioner preparation and the ordinary recurrence.
            // The seed residual handle already names the live vector produced above.
            let seed_residual = failure_handles.r;
            let audit_handles = [rigid.xr.handle(), seed_residual, rigid.p.handle(),
                rigid.rp.handle(), rigid.f.handle()];
            let prior = if audit {
                let mut bits = Vec::new();
                for handle in audit_handles {
                    bits.push(rigid_failure_read_f32(device, handle, dim)?);
                }
                Some(bits)
            } else { None };
            let reuse_domain = if !audit
                && std::env::var_os("PPF_BODY_FULL_REUSE").is_some_and(|value| value == "1")
            {
                full_body_native::ReuseDomain::new(rigid.factor_workspace_generation,
                    rigid.full_body_binding_generation, joints.dt, roots)
            } else { None };
            let mut reuse = if reuse_domain.is_some() {
                std::mem::take(&mut rigid.full_body_reuse)
            } else {
                rigid.full_body_reuse = full_body_native::Reuse::default();
                full_body_native::Reuse::default()
            };
            let compact = std::env::var_os("PPF_BODY_FULL_FACTOR_COMPACT")
                .is_some_and(|value| value == "1");
            let keep_symbolic = compact && std::env::var_os("PPF_BODY_FULL_FACTOR_SYMBOLIC_REUSE")
                .is_some_and(|value| value == "1");
            let mut symbolic = if keep_symbolic {
                std::mem::take(&mut rigid.full_body_symbolic)
            } else {
                full_body_factor::SymbolicCache::default()
            };
            let result = full_body_native::try_solve(
                device, op, reduction, joints, roots, rigid, seed_residual, work, err0, tolerance,
                auto_tree, trace, reuse_domain.map(|domain| (&mut reuse, domain)),
                compact.then_some(&mut symbolic),
            );
            if keep_symbolic {
                rigid.full_body_symbolic = symbolic;
            }
            let result = result?;
            if let Some(prior) = prior {
                for (handle, before) in audit_handles.into_iter().zip(prior) {
                    let after = rigid_failure_read_f32(device, handle, dim)?;
                    if !before.iter().zip(after).all(|(a, b)| a.to_bits() == b.to_bits()) {
                        return Err(super::scene::Fatal::invariant("full-body rejected trial altered live Krylov state"));
                    }
                }
                eprintln!("PPF_BODY_FULL_NATIVE_AUDIT original_seed_unchanged=true forced_reject=true verified={}", result.is_some());
            } else if let Some(result) = result {
                // Everything before this write is private. Export only the canonical
                // affine vector that passed the actual native residual check.
                rigid.xr.write(device, 0, &result.x)?;
                export_rigid_solve_iterate(device, reduction, rigid.xr.handle(), x, rotation_out, input)?;
                rigid.full_body_reuse = reuse;
                if let Some((started, before)) = timing_receipt {
                    let after = device.counters();
                    eprintln!("PPF_PCG_TIMING_RECEIPT input=newton coalesce_eligible={coalesce_mixed} coalesced_iterations=0 grouped_soft=false bodies={bodies} body_projection=false roots={} lock=false schwarz=false outcome=Converged iterations=0 true_residual_checks={} residual={:.9e} elapsed_ms={:.3} syncs={} dispatches={} readback_requests=unavailable readback_bytes=unavailable",
                        roots.is_some(), result.checks, result.score, started.elapsed().as_secs_f64()*1000.0,
                        after.syncs.saturating_sub(before.syncs), after.dispatches.saturating_sub(before.dispatches));
                }
                return Ok(Report { outcome: Outcome::Converged, iterations: 0, relative_residual: result.score });
            }
        } else if trace {
            eprintln!("PPF_BODY_FULL_NATIVE active=false reason=incompatible");
        }
    }

    // The direct route was declined or bypassed. Never carry an unverified
    // factor through a different solve domain or through an ordinary PCG exit.
    rigid.full_body_reuse = full_body_native::Reuse::default();
    let mut auto_resident_active = false;
    let mut lazy_tree_mode: Option<bool> = None;
    if auto_tree || std::env::var_os("PPF_BODY_TREE_PCG").as_deref()==Some(std::ffi::OsStr::new("1")) {
        let mode=std::env::var("PPF_BODY_TREE_APPLY").unwrap_or_else(|_|if auto_tree {"factor"} else {"device"}.into());
        let sparse_host = sparse_host_body_tree(device, mode == "host");
        let resident_requested=auto_tree || std::env::var_os("PPF_JOINT_RESIDENT_PCG").is_some_and(|x|x=="1");
        let factor_resident=mode=="factor" && resident_requested && joints.count!=0 && joints.can_encode_krylov()
            && (roots.is_none() || auto_tree || std::env::var_os("PPF_JOINT_RESIDENT_ROOT").is_some_and(|x|x=="1"))
            && failure_capture_dir.is_none() && std::env::var_os("PPF_PCG_TRUE_RESIDUAL_CAPTURE_DIR").is_none();
        let reason=if input!=RigidSolveInput::NewtonForce || !resident_without_hard {Some("hard_or_rhs")}
            else if lock.is_some() || reduction.map.needs_projection() || reduction.staged.reduced_schwarz.active(){Some("constraint_or_schwarz")}
            else if reduction.staged.pcg_precision != crate::args::PcgPrecision::Fp32 {Some("precision")}
            else if (resident_requested && !factor_resident) || (fused_joint_product && mode!="factor") || (coalesce_mixed && !auto_tree) || requested_mode!=0 || arithmetic_active {Some("unsupported_pcg_mode")}
            else if body_tree_scalar_extent(bodies, mode == "factor", sparse_host).is_none() {Some("body_extent")}
            else {None};
        if let Some(reason)=reason {eprintln!("PPF_BODY_TREE_PCG active=false reason={}",reason);}
        else {
            if mode!="device" && mode!="host" && mode!="factor" {eprintln!("PPF_BODY_TREE_PCG active=false reason=apply_mode");}
            else if (mode=="host" || mode=="factor") && (std::env::var_os("PPF_BODY_TREE_DEBUG_RESTART").is_some() || std::env::var_os("PPF_BODY_GROUP_SOLVE").is_some() || std::env::var_os("PPF_BODY_SCALAR64_SOLVE").is_some()) {eprintln!("PPF_BODY_TREE_PCG active=false reason=incompatible_debug");}
            else if mode=="factor" && std::env::var_os("PPF_BODY_TREE_LAZY").is_some_and(|v|v=="1") {eprintln!("PPF_BODY_TREE_PCG active=false reason=factor_lazy_unsupported");}
            else if std::env::var_os("PPF_BODY_TREE_LAZY").is_some_and(|v|v=="1") && std::env::var_os("PPF_BODY_TREE_DEBUG_RESTART").is_none() && std::env::var_os("PPF_BODY_GROUP_SOLVE").is_none() && std::env::var_os("PPF_BODY_SCALAR64_SOLVE").is_none() { lazy_tree_mode=Some(mode=="host"); }
            else {
                let reuse=mode=="factor" && resident_requested && (auto_tree || std::env::var_os("PPF_BODY_TREE_FACTOR_REUSE").is_some_and(|v|v=="1"));
                let domain=roots.map(|r|r.constraint_domain);
                let key=if reuse && domain.map_or(true,|d|d.is_cacheable()) {
                    device.instance_id().map(|instance|FactorDomainKey {device_instance:instance,workspace_generation:rigid.factor_workspace_generation,nrow:reduction.map.nrow,bodies,body_base,dim,roots:domain})
                } else {None};
                if let Some(mut cached)=rigid.body_factor_cache.take() {
                    let debt = cached.iteration_debt.filter(|debt| {
                        iteration_debt_policy && key == Some(cached.key) && debt.dt_bits == joints.dt.to_bits()
                    });
                    if key==Some(cached.key) && !debt.is_some_and(FactorIterationDebt::expired) {
                        rigid.active_factor_prepare_ms=cached.prepare_ms;
                        rigid.active_factor_borrowed=true;
                        rigid.active_factor_iteration_debt = debt;
                        rigid.body_factor=Some(cached.owner);
                        eprintln!("PPF_BODY_TREE_FACTOR_REUSE hit=true");
                    } else {
                        if let Some(debt) = debt.filter(|_| timing_receipt.is_some()) {
                            eprintln!("PPF_BODY_TREE_FACTOR_REUSE hit=false reason=iteration_debt fresh_iterations={} excess_iterations={}",
                                debt.fresh_iterations, debt.excess_iterations);
                        }
                        cached.owner.free(device)?;
                    }
                }
                if rigid.body_factor.is_none() {
                    let started=std::time::Instant::now();
                    if let Some((engine,debug,reference))=prepare_body_tree(device,op,reduction,joints,roots,mode=="host",mode=="factor")? {
                        rigid.active_factor_prepare_ms=started.elapsed().as_secs_f64()*1000.0;
                        match engine {PreparedBodyTree::Factor(owner)=>rigid.body_factor=Some(owner),PreparedBodyTree::Device(inv)=>rigid.body_inverse=Some(inv),PreparedBodyTree::Host(host)=>rigid.body_host_tree=Some(host)}
                        if iteration_debt_policy && key.is_some() && rigid.body_factor.is_some() {
                            rigid.active_factor_fresh_prefix = Some(0);
                        }
                        rigid.body_inverse_debug=debug;rigid.body_group_reference=reference;
                    }
                }
                if rigid.body_factor.is_some() || rigid.body_inverse.is_some() || rigid.body_host_tree.is_some() {
                    rigid.active_factor_key=key;
                    auto_resident_active = runtime_auto_policy::automatic_active(auto_tree, rigid.body_factor.is_some());
                    if auto_resident_active { coalesce_mixed = false; }
                    eprintln!("PPF_BODY_TREE_PCG active=true apply={}",mode);
                }
            }
        }
    }

    if rigid.body_inverse.is_some() || rigid.body_host_tree.is_some() || rigid.body_factor.is_some() {
        rigid.body_seed_x.size(device,dim,ppf_cts_compute::AllocLabel("body_tree.seed_x"))?;
        rigid.body_seed_r.size(device,dim,ppf_cts_compute::AllocLabel("body_tree.seed_r"))?;
        device.copy(rigid.body_seed_x.handle(),0,rigid.xr.handle(),0,dim*4)?;
        device.copy(rigid.body_seed_r.handle(),0,rigid.r.handle(),0,dim*4)?;
        if let Some(ref owner)=rigid.body_factor {device.run("body_factor.clear_fault",|encoder|owner.encode_clear_fault(encoder))?;}
    }

    // One selected frozen mixed solve only; never commits the candidate operator.
    if std::env::var_os("PPF_REDUCED_OPERATOR_SHADOW").is_some_and(|v|v=="1") {
        static COUNT:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);
        let solve=COUNT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)+1;
        let selected=std::env::var("PPF_REDUCED_OPERATOR_SOLVE").ok().and_then(|v|v.parse::<usize>().ok()).unwrap_or(1);
        if solve==selected {
            if input!=RigidSolveInput::NewtonForce || !resident_without_hard || lock.is_some() || reduction.map.needs_projection() || reduction.staged.reduced_schwarz.active() || rigid.body_inverse.is_some() || rigid.body_host_tree.is_some() || rigid.body_factor.is_some() || dim%3!=0 {
                return Err(Fault::Platform{call:"reduced operator shadow",detail:"unsupported domain; no commit".into()}.into());
            }
            let before=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;
            if before.iter().flatten().any(|&v|v!=0){return Err(super::scene::Fatal::invariant("reduced operator preexisting fault"));}
            macro_rules! reduced_fault_gate {($stage:expr)=>{{let now=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;if now!=before||now.iter().flatten().any(|&v|v!=0){return Err(super::scene::Fatal::invariant(format!("reduced shadow fault changed: {}",$stage)));}}};}
            let r0=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;let x0=rigid_failure_read_f32(device,rigid.xr.handle(),dim)?;
            let start=std::time::Instant::now();let blocks=super::reduced_operator_capture::capture(device,op,reduction,joints)?;let capture_ms=start.elapsed().as_secs_f64()*1000.;reduced_fault_gate!("capture");
            let start=std::time::Instant::now();let directed:Vec<_>=blocks.into_iter().map(|b|super::reduced_operator_pack::DirectedBlock{row:b.row,col:b.col,value:b.value}).collect();
            let packed=super::reduced_operator_pack::pack((dim/3) as u32,&directed).map_err(super::scene::Fatal::invariant)?;let pack_ms=start.elapsed().as_secs_f64()*1000.;
            let start=std::time::Instant::now();let mut owner=super::reduced_operator_device::ReducedOperatorDevice::upload(device,&packed)?;let upload_ms=start.elapsed().as_secs_f64()*1000.;
            let mut sin=Buffer::<f32>::default();let mut a=Buffer::<f32>::default();let mut b=Buffer::<f32>::default();
            let attempt=(||->super::scene::FatalResult<()> {
                sin.size(device,dim,ppf_cts_compute::AllocLabel("reduced.shadow.in"))?;a.size(device,dim,ppf_cts_compute::AllocLabel("reduced.shadow.native"))?;b.size(device,dim,ppf_cts_compute::AllocLabel("reduced.shadow.cached"))?;
                for kind in 0..3 {let mut x=vec![0f32;dim];for i in 0..dim {if (kind==0&&i<body_base)||(kind==1&&i>=body_base)||kind==2 {x[i]=(((i*17+3)%31) as f32-15.)*0.03125;}}
                    sin.write(device,0,&x)?;project_tangent!(sin.handle());apply_reduced!(sin.handle(),a.handle(),true);owner.apply(device,sin.handle(),b.handle())?;project_tangent!(b.handle());
                    let av=rigid_failure_read_f32(device,a.handle(),dim)?;let bv=rigid_failure_read_f32(device,b.handle(),dim)?;let mut max=0f64;let mut norm=0f64;let mut delta=0f64;let mut orig=0f64;
                    for (&u,&v) in av.iter().zip(&bv){if !u.is_finite()||!v.is_finite(){return Err(super::scene::Fatal::invariant("reduced probe nonfinite"));}let e=u as f64-v as f64;max=max.max(e.abs());norm=norm.max((u as f64).abs());delta+=e*e;orig+=(u as f64)*(u as f64);}
                    let mut groups=Vec::with_capacity(1+bodies);let mut ranges=Vec::with_capacity(1+bodies);ranges.push((0,body_base));for body in 0..bodies{ranges.push((body_base+6*body,body_base+6*body+6));}for (lo,hi) in ranges{let mut native=0f64;let mut error=0f64;for i in lo..hi{native+=(av[i] as f64).abs();error+=(av[i] as f64-bv[i] as f64).abs();}groups.push((native,error));}
                    reduced_fault_gate!("probe");
                    eprintln!("PPF_REDUCED_PROBE solve={} kind={} max_abs={:.9e} max_relative={:.9e} frobenius_relative={:.9e} groups_native_l1_error_l1={:?}",solve,kind,max,if norm>0.{max/norm}else{max},if orig>0.{(delta/orig).sqrt()}else{delta.sqrt()},groups);
                }
                let cap=std::env::var("PPF_REDUCED_OPERATOR_CAP").ok().and_then(|v|v.parse::<u32>().ok()).unwrap_or(1000).min(max_iterations);
                for mode in 0..2 {let t=std::time::Instant::now();let report=super::shadow_tree::shadow_pcg(&r0,&rigid.group_initial,body_base,tolerance,cap,|kind,values|{
                    let run=(||->super::scene::FatalResult<(Vec<f32>,f32)>{match kind {
                      super::shadow_tree::Call::Apply=>{sin.write(device,0,values)?;project_tangent!(sin.handle());if mode==0{apply_reduced!(sin.handle(),a.handle(),true);}else{owner.apply(device,sin.handle(),a.handle())?;project_tangent!(a.handle());}Ok((rigid_failure_read_f32(device,a.handle(),dim)?,0.))},
                      super::shadow_tree::Call::Precond=>{sin.write(device,0,values)?;reduction.apply_precond(device,factor,inverse_diagonal,sin.handle(),a.handle())?;project_tangent!(a.handle());Ok((rigid_failure_read_f32(device,a.handle(),dim)?,0.))},
                      super::shadow_tree::Call::Verify=>{let trial:Vec<f32>=x0.iter().zip(values).map(|(a,b)|a+b).collect();sin.write(device,0,&trial)?;project_tangent!(sin.handle());if let Some(root)=roots{root.particular(device,joints.state,sin.handle())?;}apply_reduced!(sin.handle(),a.handle(),false);combine(device,rigid.f.handle(),a.handle(),a.handle(),1.,-1.,dim)?;project_tangent!(a.handle());group_l1_device(device,a.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;let score=super::pdrd::worst_relative_residual(&rigid.group_current,&rigid.group_initial,err0);Ok((rigid_failure_read_f32(device,a.handle(),dim)?,score))}
                    }} )();run.map_err(|e|e.detail)
                });reduced_fault_gate!("pcg_mode");eprintln!("PPF_REDUCED_SHADOW solve={} mode={} capture_ms={:.6} pack_ms={:.6} upload_ms={:.6} run_ms={:.6} directed_blocks={} result={:?} scope=host_f64dot_f32vector",solve,if mode==0{"native"}else{"cached"},capture_ms,pack_ms,upload_ms,t.elapsed().as_secs_f64()*1000.,directed.len(),report);if let Err(e)=report{return Err(super::scene::Fatal::invariant(format!("reduced shadow failed: {e}")));}}
                Ok(())
            })();
            let end_fault=super::shadow_tree::fault_snapshot(device,op,joints);let free_owner=owner.free(device);let free_in=sin.free(device);let free_a=a.free(device);let free_b=b.free(device);
            attempt?;free_owner?;free_in?;free_a?;free_b?;
            let after=end_fault.map_err(super::scene::Fatal::invariant)?;if before!=after||after.iter().flatten().any(|&v|v!=0){return Err(super::scene::Fatal::invariant("reduced shadow fault changed"));}
            return Err(Fault::Platform{call:"reduced operator shadow",detail:"comparison finished; intentional no-commit".into()}.into());
        }
    }

    // Explicit diagnostic-only shadow: process refuses the physical step after
    // comparison. Shared operator temporaries can therefore never be committed.
    if std::env::var_os("PPF_TREE_SHADOW").as_deref()==Some(std::ffi::OsStr::new("1")) {
        static SHADOW_COUNT:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);
        let n=SHADOW_COUNT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)+1;
        let selected=std::env::var("PPF_TREE_SHADOW_SOLVE").ok().and_then(|s|s.parse::<usize>().ok()).unwrap_or(1);
        if n==selected {
            if input!=RigidSolveInput::NewtonForce || lock.is_some() || reduction.map.needs_projection() || reduction.staged.reduced_schwarz.active() || !resident_without_hard {
                return Err(Fault::Platform{call:"tree shadow diagnostic",detail:"unsupported solve domain; no physical commit".into()}.into());
            }
            let diag_dir=std::env::var_os("PPF_TREE_SHADOW_DIAG_DIR").map(std::path::PathBuf::from);
            if diag_dir.is_some() && (std::env::var_os("PPF_TREE_DENSE_SHADOW").as_deref()!=Some(std::ffi::OsStr::new("1")) || std::env::var_os("PPF_TREE_DIRECT_USE").is_some() || op.has_tet_factor() || joints.soft.is_some()) {
                return Err(super::scene::Fatal::invariant("shadow residual diagnostic requires dense, no direct use, no tet and no soft"));
            }
            let started=std::time::Instant::now();
            let blocks=rigid.failure_precond_blocks.ok_or_else(||super::scene::Fatal::invariant("shadow missing assembled BJ blocks"))?;
            let tree=super::shadow_tree::capture(device,blocks,bodies,roots,joints).map_err(|e|super::scene::Fatal::invariant(format!("tree shadow setup refused: {e}")))?;
            let setup_ms=started.elapsed().as_secs_f64()*1000.;
            let r0=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
            let x0=rigid_failure_read_f32(device,rigid.xr.handle(),dim)?;
            let diag_f=if diag_dir.is_some(){Some(rigid_failure_read_f32(device,rigid.f.handle(),dim)?)}else{None};
            let mut sin=ppf_cts_compute::Buffer::<f32>::default();let mut sout=ppf_cts_compute::Buffer::<f32>::default();
            sin.size(device,dim,ppf_cts_compute::AllocLabel("shadow.input"))?;
            sout.size(device,dim,ppf_cts_compute::AllocLabel("shadow.output"))?;
            let dense=if std::env::var_os("PPF_TREE_DENSE_SHADOW").as_deref()==Some(std::ffi::OsStr::new("1")) {
                let free=tree.free_indices();let m=free.len();
                if m==0||m>1024||m.checked_mul(m).is_none() {return Err(super::scene::Fatal::invariant("dense shadow refused dimension >1024 or overflow"));}
                let before=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;
                if before.iter().flatten().any(|&v|v!=0) {return Err(super::scene::Fatal::invariant("dense shadow preexisting fault"));}
                let capture_start=std::time::Instant::now();let mut columns=Vec::with_capacity(m*m);let mut basis=vec![0f32;dim];
                for &slot in &free {basis[body_base+slot]=1.;sin.write(device,0,&basis)?;basis[body_base+slot]=0.;
                    project_tangent!(sin.handle());apply_reduced!(sin.handle(),sout.handle(),true);
                    let product=rigid_failure_read_f32(device,sout.handle(),dim)?;
                    for &row in &free {columns.push(product[body_base+row]);}
                }
                if std::env::var_os("PPF_TREE_DIRECT_SHADOW").as_deref()==Some(std::ffi::OsStr::new("1")) {
                    let t=std::time::Instant::now();let direct=super::body_block_capture::capture(device,op,reduction,joints,&free)?;
                    let mut max=0f64;let mut norm=0f64;let mut sq=0f64;let mut refsq=0f64;
                    for (&a,&b) in direct.iter().zip(&columns){let d=a as f64-b as f64;max=max.max(d.abs());norm=norm.max((b as f64).abs());sq+=d*d;refsq+=(b as f64)*(b as f64);}
                    eprintln!("PPF_TREE_DIRECT_COMPARE elapsed_ms={:.6} max_abs={:.9e} max_relative={:.9e} frobenius_relative={:.9e}",t.elapsed().as_secs_f64()*1000.,max,if norm>0.{max/norm}else{max},if refsq>0.{(sq/refsq).sqrt()}else{sq.sqrt()});
                    if std::env::var_os("PPF_TREE_DIRECT_USE").as_deref()==Some(std::ffi::OsStr::new("1")){columns=direct;eprintln!("PPF_TREE_DIRECT_USE active");}
                }
                if std::env::var_os("PPF_BODY_INVERSE_SELF_CHECK").as_deref()==Some(std::ffi::OsStr::new("1")){super::body_inverse_device::self_check(device)?;eprintln!("PPF_BODY_INVERSE_SELF_CHECK passed");}
                let capture_ms=capture_start.elapsed().as_secs_f64()*1000.;
                let after=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;
                if before!=after || after.iter().flatten().any(|&v|v!=0){return Err(super::scene::Fatal::invariant("dense shadow invalid: capture fault changed"));}
                let mut raw_skew=0f64;let mut raw_norm=0f64;for i in 0..m {for j in 0..m {let a=columns[j*m+i] as f64;let b=columns[i*m+j] as f64;if !a.is_finite(){return Err(super::scene::Fatal::invariant("dense shadow nonfinite block"));}raw_skew=raw_skew.max((a-b).abs());raw_norm=raw_norm.max(a.abs());}}
                eprintln!("PPF_TREE_DENSE_INPUT free_dof={} capture_ms={:.6} skew_abs={:.9e} block_abs_max={:.9e} skew_ratio={:.9e} matrix_bytes={}",m,capture_ms,raw_skew,raw_norm,if raw_norm>0.{raw_skew/raw_norm}else{0.},m*m*4);
                if let Some(dir)=diag_dir.as_ref() {
                    use self::shadow_residual_diagnostics as diag;
                    let f=diag_f.as_ref().unwrap();
                    for v in [&columns[..],&x0[..],&r0[..],&f[..],&rigid.group_initial[..]] {diag::finite(v,1024*1024).map_err(super::scene::Fatal::invariant)?;}
                    if dim>1024 || !err0.is_finite() || !tolerance.is_finite() || !joints.start_time_s.is_finite() || !joints.sample_time_s.is_finite() || !joints.dt.is_finite(){return Err(super::scene::Fatal::invariant("shadow diagnostic extent or scalar invalid"));}
                    let before=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;
                    if before.iter().flatten().any(|&v|v!=0){return Err(super::scene::Fatal::invariant("shadow diagnostic preexisting fault"));}
                    let zero=vec![0f32;dim];sin.write(device,0,&zero)?;
                    apply_reduced!(sin.handle(),sout.handle(),false);let o0=rigid_failure_read_f32(device,sout.handle(),dim)?;
                    apply_reduced!(sin.handle(),sout.handle(),true);let k0=rigid_failure_read_f32(device,sout.handle(),dim)?;
                    sin.write(device,0,&x0)?;apply_reduced!(sin.handle(),sout.handle(),false);let ox0=rigid_failure_read_f32(device,sout.handle(),dim)?;
                    combine(device,rigid.f.handle(),sout.handle(),sout.handle(),1.,-1.,dim)?;project_tangent!(sout.handle());
                    let recomputed=rigid_failure_read_f32(device,sout.handle(),dim)?;
                    group_l1_device(device,sout.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;let groups=rigid.group_current.clone();
                    apply_reduced!(sin.handle(),sout.handle(),true);let kx0=rigid_failure_read_f32(device,sout.handle(),dim)?;
                    apply_reduced!(sin.handle(),sout.handle(),false);let repeat=rigid_failure_read_f32(device,sout.handle(),dim)?;
                    combine(device,rigid.f.handle(),sout.handle(),sout.handle(),1.,-1.,dim)?;project_tangent!(sout.handle());let repeat_r=rigid_failure_read_f32(device,sout.handle(),dim)?;
                    let after=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;
                    if before!=after || after.iter().flatten().any(|&v|v!=0){return Err(super::scene::Fatal::invariant("shadow diagnostic cold fault"));}
                    for v in [&o0[..],&k0[..],&ox0[..],&kx0[..],&repeat[..],&repeat_r[..],&recomputed[..],&groups[..]] {diag::finite(v,1024*1024).map_err(super::scene::Fatal::invariant)?;}
                    diag::fresh(dir).map_err(super::scene::Fatal::invariant)?;
                    let slots:Vec<usize>=free.iter().map(|&i|body_base+i).collect();
                    let cap=std::env::var("PPF_TREE_SHADOW_CAP").ok().and_then(|s|s.parse::<u32>().ok()).unwrap_or(1000).min(max_iterations);
                    diag::write(dir,"cold.json",&serde_json::json!({"solve":n,"start":joints.start_time_s,"sample":joints.sample_time_s,"dt":joints.dt,"tol":tolerance,"cap":cap,"dim":dim,"body_base":body_base,"bodies":bodies,"root_count":roots.map_or(0,|root|root.count),"frozen_flags":{"dense":true,"direct_use":false,"tet":false,"soft":false,"frozen_ball":joints.frozen_ball.is_some(),"frozen_angular":joints.frozen_angular.is_some(),"frozen_angular_tangent":joints.frozen_angular_tangent.is_some(),"frozen_control_tangent":joints.frozen_control_tangent.is_some(),"frozen_extension_tangent":joints.frozen_extension_tangent.is_some()},"group_initial":rigid.group_initial,"err0":err0,"x0":x0,"r0":r0,"f":f,"columns":columns,"columns_layout":"column_major","free_reduced_slots":slots,"aorig0":o0,"akry0":k0,"aorig_x0":ox0,"akry_x0":kx0,"aorig_x0_repeat":repeat,"r0_recomputed_repeat":repeat_r,"r0_recomputed":recomputed,"r0_groups":groups})).map_err(super::scene::Fatal::invariant)?;
                }
                let factor_start=std::time::Instant::now();
                let (factor,skew,norm)=super::shadow_tree::DenseBody::factor(free,bodies,&columns).map_err(|e|super::scene::Fatal::invariant(format!("dense shadow SPD refused: {e}")))?;
                eprintln!("PPF_TREE_DENSE_CAPTURE free_dof={} capture_ms={:.6} factor_ms={:.6} skew_abs={:.9e} block_abs_max={:.9e} skew_ratio={:.9e} spd=accepted max_dof=1024 matrix_bytes={}",m,capture_ms,factor_start.elapsed().as_secs_f64()*1000.,skew,norm,if norm>0.{skew/norm}else{0.},m*m*4);
                let p1_start=std::time::Instant::now();let p1=tree.from_dense(&columns).map_err(|e|super::scene::Fatal::invariant(format!("real-block forest SPD refused: {}",e.0)))?;
                let p1_ms=p1_start.elapsed().as_secs_f64()*1000.;
                eprintln!("PPF_TREE_REAL_BLOCK_SETUP factor_ms={:.6} spd=accepted source=probed_body_and_unique_ball_forest",p1_ms);
                let p2_start=std::time::Instant::now();let p2=match tree.from_dense_weighted(&columns) {Ok(t)=>Some(t),Err(e)=>{eprintln!("PPF_TREE_WEIGHTED_SETUP spd=refused reason={:?}",e.0);None}};
                let p2_ms=p2_start.elapsed().as_secs_f64()*1000.;if let Some(t)=p2.as_ref(){eprintln!("PPF_TREE_WEIGHTED_SETUP spd=accepted setup_ms={:.6} omitted_cycles={}",p2_ms,t.omitted_cycles);}
                let save=std::env::var_os("PPF_TREE_LAGGED_SAVE");let load=std::env::var_os("PPF_TREE_LAGGED_LOAD");
                if save.is_some()&&load.is_some(){return Err(super::scene::Fatal::invariant("lagged shadow save/load mutually exclusive"));}
                if load.is_some() && p2.is_none(){return Err(super::scene::Fatal::invariant("lagged comparison requires fresh weighted SPD"));}
                let lagged=if save.is_some()||load.is_some() {
                    let domain=super::shadow_tree::domain(reduction.map,joints,tree.free_indices()).map_err(super::scene::Fatal::invariant)?;
                    if let Some(path)=save {super::shadow_tree::save_lagged(std::path::Path::new(&path),domain,columns.clone(),n,joints).map_err(super::scene::Fatal::invariant)?;eprintln!("PPF_TREE_LAGGED_SAVE source_solve={} start={:.17e} sample={:.9e}",n,joints.start_time_s,joints.sample_time_s);None}
                    else {let path=load.unwrap();let (old,source,old_start,old_sample)=super::shadow_tree::load_lagged(std::path::Path::new(&path),&domain).map_err(super::scene::Fatal::invariant)?;
                        let t=std::time::Instant::now();let lag=tree.from_dense_weighted(&old).map_err(|e|super::scene::Fatal::invariant(format!("lagged tree SPD refused: {}",e.0)))?;let ms=t.elapsed().as_secs_f64()*1000.;
                        eprintln!("PPF_TREE_LAGGED_LOAD source_solve={} source_start={:.17e} source_sample={:.9e} current_solve={} current_start={:.17e} current_sample={:.9e} setup_ms={:.6} spd=accepted",source,old_start,old_sample,n,joints.start_time_s,joints.sample_time_s,ms);Some((lag,ms))}
                } else {None};
                Some((factor,p1,p1_ms,p2,p2_ms,lagged))
            } else {None};
            let shadow_cap=std::env::var("PPF_TREE_SHADOW_CAP").ok().and_then(|s|s.parse::<u32>().ok()).unwrap_or(1000).min(max_iterations);
            let compare_apply=std::env::var_os("PPF_TREE_SHADOW_APPLY_COMPARE").is_some_and(|x|x=="1");
            let mut shadow_device=None;let mut shadow_device_setup=0f64;
            if compare_apply {
                let weighted=dense.as_ref().and_then(|d|d.3.as_ref()).ok_or_else(||super::scene::Fatal::invariant("apply compare needs weighted SPD"))?;
                let t=std::time::Instant::now();let packed=pack_shadow_tree(weighted,bodies).map_err(super::scene::Fatal::invariant)?;
                shadow_device=Some(super::body_inverse_device::BodyInverseDevice::upload(device,&packed)?);shadow_device_setup=t.elapsed().as_secs_f64()*1000.;
            }
            let mut modes:Vec<usize>=(0..if dense.as_ref().is_some_and(|d|d.5.is_some()){6}else if dense.as_ref().is_some_and(|d|d.3.is_some()){5}else if dense.is_some(){4}else{2}).collect();
            if compare_apply {modes.extend([6,7]);}
            if let Ok(requested)=std::env::var("PPF_TREE_SHADOW_MODES") {modes=shadow_residual_diagnostics::modes(&requested,&modes).map_err(super::scene::Fatal::invariant)?;}
            let mut outcomes=Vec::new();
            let shadow_result=(|| -> super::scene::FatalResult<()> {
            for mode in modes {
                let fault_before=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;
                let fault_before_count=fault_before.iter().flatten().filter(|&&v|v!=0).count();
                if fault_before_count!=0 {return Err(super::scene::Fatal::invariant(format!("tree shadow invalid: preexisting faults={fault_before_count}")));}
                let run_start=std::time::Instant::now();let mut pre_ms=0f64;let mut pre_calls=0u32;let(mut host_read_ms,mut host_solve_ms,mut host_write_ms,mut host_bytes)=(0f64,0f64,0f64,0u64);
                let mut verify_count=0usize;
                let report=super::shadow_tree::shadow_pcg(&r0,&rigid.group_initial,body_base,tolerance,shadow_cap,|kind,values| {
                    let op=(|| -> super::scene::FatalResult<(Vec<f32>,f32)> {
                        match kind {
                            super::shadow_tree::Call::Apply=>{
                                sin.write(device,0,values)?;project_tangent!(sin.handle());apply_reduced!(sin.handle(),sout.handle(),true);
                                Ok((rigid_failure_read_f32(device,sout.handle(),dim)?,0.))
                            }
                            super::shadow_tree::Call::Precond=>{
                                let t=std::time::Instant::now();sin.write(device,0,values)?;
                                reduction.apply_precond(device,factor,inverse_diagonal,sin.handle(),sout.handle())?;
                                project_tangent!(sout.handle());let bj=rigid_failure_read_f32(device,sout.handle(),dim)?;
                                let out=match mode {6=>{shadow_device.as_ref().unwrap().apply(device,sin.span(body_base,6*bodies),sout.span(body_base,6*bodies))?;project_tangent!(sout.handle());rigid_failure_read_f32(device,sout.handle(),dim)?},7=>{let(r,s,w,b)=host_tree_apply(dense.as_ref().unwrap().3.as_ref().unwrap(),device,sin.handle(),sout.handle(),body_base,bodies)?;host_read_ms+=r;host_solve_ms+=s;host_write_ms+=w;host_bytes+=b;project_tangent!(sout.handle());rigid_failure_read_f32(device,sout.handle(),dim)?},1=>tree.apply_mixed(values,&bj,body_base).map_err(|e|super::scene::Fatal::invariant(e.0))?,2=>dense.as_ref().unwrap().0.apply_mixed(values,&bj,body_base).map_err(super::scene::Fatal::invariant)?,3=>dense.as_ref().unwrap().1.apply_mixed(values,&bj,body_base).map_err(|e|super::scene::Fatal::invariant(e.0))?,4=>dense.as_ref().unwrap().3.as_ref().unwrap().apply_mixed(values,&bj,body_base).map_err(|e|super::scene::Fatal::invariant(e.0))?,5=>dense.as_ref().unwrap().5.as_ref().unwrap().0.apply_mixed(values,&bj,body_base).map_err(|e|super::scene::Fatal::invariant(e.0))?,_=>bj};
                                pre_ms+=t.elapsed().as_secs_f64()*1000.;pre_calls+=1;Ok((out,0.))
                            }
                            super::shadow_tree::Call::Verify=>{
                                let trial:Vec<f32>=x0.iter().zip(values).map(|(a,b)|a+b).collect();sin.write(device,0,&trial)?;
                                project_tangent!(sin.handle());if let Some(root)=roots {root.particular(device,joints.state,sin.handle())?;}
                                verify_count+=1;
                                let capture=diag_dir.is_some() && verify_count<=2;
                                let actual=if capture {Some(rigid_failure_read_f32(device,sin.handle(),dim)?)}else{None};
                                apply_reduced!(sin.handle(),sout.handle(),false);
                                let orig=if capture {Some(rigid_failure_read_f32(device,sout.handle(),dim)?)}else{None};
                                combine(device,rigid.f.handle(),sout.handle(),sout.handle(),1.,-1.,dim)?;
                                project_tangent!(sout.handle());group_l1_device(device,sout.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
                                let score=super::pdrd::worst_relative_residual(&rigid.group_current,&rigid.group_initial,err0);
                                let native=rigid_failure_read_f32(device,sout.handle(),dim)?;
                                if capture {
                                    apply_reduced!(sin.handle(),sout.handle(),true);let kry=rigid_failure_read_f32(device,sout.handle(),dim)?;
                                    let actual=actual.unwrap();let orig=orig.unwrap();
                                    for v in [&actual[..],&orig[..],&kry[..],&native[..],values,&trial[..],&rigid.group_current[..]] {shadow_residual_diagnostics::finite(v,1024*1024).map_err(super::scene::Fatal::invariant)?;}
                                    if !score.is_finite(){return Err(super::scene::Fatal::invariant("nonfinite shadow score"));}
                                    let delta:Vec<f64>=actual.iter().zip(&x0).map(|(&a,&b)|a as f64-b as f64).collect();
                                    let name=format!("verify_{}_{}.json",match mode{0=>"bj",1=>"forest",2=>"dense",3=>"real_block_forest",4=>"weighted_forest",5=>"lagged_weighted",6=>"device_weighted",7=>"host_weighted",_=>"unknown"},verify_count);
                                    shadow_residual_diagnostics::write(diag_dir.as_ref().unwrap(),&name,&serde_json::json!({"delta_accumulated":values,"trial_rounded":trial,"actual_x":actual,"delta_actual_f64":delta,"aorig_x":orig,"akry_x":kry,"r_native":native,"groups":rigid.group_current,"score":score})).map_err(super::scene::Fatal::invariant)?;
                                }
                                Ok((native,score))
                            }
                        }
                    })();op.map_err(|e|e.detail)
                });
                let fault_after=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;
                let fault_after_count=fault_after.iter().flatten().filter(|&&v|v!=0).count();
                eprintln!("PPF_TREE_SHADOW mode={} solve={} setup_ms={:.6} apply_calls={} apply_ms={:.6} total_ms={:.6} result={:?} omitted_cycles={} fault_before={} fault_after={} scope=host_f64dot_f32vector",match mode{1=>"forest",2=>"dense",3=>"real_block_forest",4=>"weighted_forest",5=>"lagged_weighted",6=>"device_weighted",7=>"host_weighted",_=>"bj"},n,if mode==1{setup_ms}else if mode==3{dense.as_ref().unwrap().2}else if mode==4{dense.as_ref().unwrap().4}else if mode==5{dense.as_ref().unwrap().5.as_ref().unwrap().1}else if mode==6{shadow_device_setup+dense.as_ref().unwrap().4}else if mode==7{dense.as_ref().unwrap().4}else{0.},pre_calls,pre_ms,run_start.elapsed().as_secs_f64()*1000.,report,if mode==4{dense.as_ref().unwrap().3.as_ref().unwrap().omitted_cycles}else if mode==5{dense.as_ref().unwrap().5.as_ref().unwrap().0.omitted_cycles}else if mode==6 || mode==7{dense.as_ref().unwrap().3.as_ref().unwrap().omitted_cycles}else{tree.omitted_cycles},fault_before_count,fault_after_count);
                if mode==7 {eprintln!("PPF_TREE_SHADOW_HOST_TRANSFER calls={} bytes={} read_ms={:.6} solve_ms={:.6} write_ms={:.6} common_shadow_full_vector_io=true native_wall=false",pre_calls,host_bytes,host_read_ms,host_solve_ms,host_write_ms);}
                if fault_before!=fault_after || fault_after_count!=0 {return Err(super::scene::Fatal::invariant("tree shadow invalid: shared fault changed; no comparison"));}
                outcomes.push(serde_json::json!({"mode":mode,"verify_count":verify_count,"result":format!("{:?}",report)}));
                if let Err(error)=report {if !shadow_residual_diagnostics::cap_allowed(&error,std::env::var_os("PPF_TREE_SHADOW_CONTINUE_CAP").as_deref()==Some(std::ffi::OsStr::new("1"))) {return Err(super::scene::Fatal::invariant(format!("tree shadow invalid: {error}")));}}
            }
            if let Some(dir)=diag_dir.as_ref(){shadow_residual_diagnostics::write(dir,"outcomes.json",&serde_json::json!(outcomes)).map_err(super::scene::Fatal::invariant)?;}
            Ok(())})();
            let cleanup=if let Some(mut owner)=shadow_device.take(){owner.free(device)}else{Ok(())};
            shadow_result?;cleanup?;
            return Err(Fault::Platform{call:"tree shadow diagnostic",detail:"comparison finished; intentional no-commit stop".into()}.into());
        }
    }
    // THE REDUCED WORKSPACE'S OWN TERMS, not the main one's: the fold's extent
    // is its buffer's length, so folding with the vertex-sized pair would read
    // past the end of every reduced vector here.
    if warm_eligible {
        combine(device, rigid.xr.handle(), rigid.xr.handle(), rigid.warm_seed.handle(), 1.0, 0.0, dim)?;
    }
    let RigidWorkspace { product, magnitude, .. } = rigid;
    let mut terms = DotTerms { product, magnitude };
    let mut warm_accepted = false;
    let mut warm_verified = f32::INFINITY;
    if warm_has_history && requested_mode == 1 {
        // r0 -> z. Candidate s+h uses the *affine* operator; subtract A(s)
        // from A(s+h) so constant affine terms never contaminate A*h.
        combine(device, rigid.r.handle(), rigid.r.handle(), rigid.z.handle(), 1.0, 0.0, dim)?;
        combine(device, rigid.xr.handle(), rigid.warm_history.handle(), rigid.p.handle(), 1.0, 1.0, dim)?;
        apply_reduced!(rigid.p.handle(), rigid.r.handle(), false);
        warm_difference(device, rigid.r.handle(), rigid.rp.handle(), rigid.p.handle(), dim)?;
        let numerator = terms.dot(device, rigid.p.handle(), rigid.z.handle(), dim, &work.fold, &mut work.scalar)?;
        let curvature = terms.dot(device, rigid.warm_history.handle(), rigid.p.handle(), dim, &work.fold, &mut work.scalar)?;
        let denominator = terms.dot(device, rigid.p.handle(), rigid.p.handle(), dim, &work.fold, &mut work.scalar)?;
        let alpha = warm_start_policy::alpha(numerator,denominator,curvature);
        if let Some(alpha) = alpha {
            combine(device, rigid.xr.handle(), rigid.warm_history.handle(), rigid.p.handle(), 1.0, alpha, dim)?;
            apply_reduced!(rigid.p.handle(), rigid.rp.handle(), false);
            combine(device, rigid.f.handle(), rigid.rp.handle(), rigid.r.handle(), 1.0, -1.0, dim)?;
            project_tangent!(rigid.r.handle());
            group_l1_device(device, rigid.r.handle(), &mut rigid.group_norm, &work.fold,
                body_base, bodies, &mut rigid.group_current)?;
            warm_verified = super::pdrd::worst_relative_residual(&rigid.group_current, &rigid.group_initial, err0);
            let cold = super::pdrd::worst_relative_residual(&rigid.group_initial, &rigid.group_initial, err0);
            if warm_start_policy::accept(warm_verified,cold) {
                combine(device, rigid.p.handle(), rigid.p.handle(), rigid.xr.handle(), 1.0, 0.0, dim)?;
                warm_accepted = true;
            }
        }
        if !warm_accepted {
            combine(device, rigid.z.handle(), rigid.z.handle(), rigid.r.handle(), 1.0, 0.0, dim)?;
        }
        eprintln!("PPF_PCG_WARM_START {} alpha={:?} residual={:.9e}",
            if warm_accepted {"accepted"} else {"rejected"}, alpha, warm_verified);
    } else if warm_has_history && auto_candidate {
        // z is the untouched cold residual, rp is A_affine(cold), xr stays
        // cold until after the full candidate passes the original operator.
        combine(device, rigid.r.handle(), rigid.r.handle(), rigid.z.handle(), 1.0, 0.0, dim)?;
        combine(device, rigid.xr.handle(), rigid.warm_history.handle(), rigid.p.handle(), 1.0, 1.0, dim)?;
        apply_reduced!(rigid.p.handle(), rigid.r.handle(), false);
        combine(device, rigid.r.handle(), rigid.rp.handle(), rigid.warm_ad1.handle(), 1.0, -1.0, dim)?;
        let g11=terms.dot(device,rigid.warm_history.handle(),rigid.warm_ad1.handle(),dim,&work.fold,&mut work.scalar)?;
        let b1=terms.dot(device,rigid.warm_history.handle(),rigid.z.handle(),dim,&work.fold,&mut work.scalar)?;
        let mut g12=0.0; let mut g21=0.0; let mut g22=0.0; let mut b2=0.0;
        if usable_warm_count >= 2 {
            combine(device, rigid.xr.handle(), rigid.warm_history2.handle(), rigid.p.handle(), 1.0, 1.0, dim)?;
            apply_reduced!(rigid.p.handle(), rigid.r.handle(), false);
            combine(device, rigid.r.handle(), rigid.rp.handle(), rigid.warm_ad2.handle(), 1.0, -1.0, dim)?;
            g12=terms.dot(device,rigid.warm_history.handle(),rigid.warm_ad2.handle(),dim,&work.fold,&mut work.scalar)?;
            g21=terms.dot(device,rigid.warm_history2.handle(),rigid.warm_ad1.handle(),dim,&work.fold,&mut work.scalar)?;
            g22=terms.dot(device,rigid.warm_history2.handle(),rigid.warm_ad2.handle(),dim,&work.fold,&mut work.scalar)?;
            b2=terms.dot(device,rigid.warm_history2.handle(),rigid.z.handle(),dim,&work.fold,&mut work.scalar)?;
        }
        let coeff=warm_start_policy::energy_coefficients(g11,g12,g21,g22,b1,b2,usable_warm_count>=2);
        let mut delta_energy=f32::NAN;
        if let Some((c1,c2,rank))=coeff {
            combine(device,rigid.xr.handle(),rigid.warm_history.handle(),rigid.p.handle(),1.0,c1,dim)?;
            let candidate = if rank==2 {
                warm_add_second(device,rigid.p.handle(),rigid.warm_history2.handle(),rigid.warm_ad2.handle(),c2,dim)?;
                rigid.warm_ad2.handle()
            } else { rigid.p.handle() };
            apply_reduced!(candidate,rigid.rp.handle(),false);
            combine(device,rigid.f.handle(),rigid.rp.handle(),rigid.r.handle(),1.0,-1.0,dim)?;
            project_tangent!(rigid.r.handle());
            group_l1_device(device,rigid.r.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
            warm_verified=super::pdrd::worst_relative_residual(&rigid.group_current,&rigid.group_initial,err0);
            // Delta is the actual rounded candidate minus cold. The difference
            // of current original residuals includes the true affine A*delta.
            combine(device,candidate,rigid.xr.handle(),rigid.rp.handle(),1.0,-1.0,dim)?;
            let dr0=terms.dot(device,rigid.rp.handle(),rigid.z.handle(),dim,&work.fold,&mut work.scalar)?;
            let dr=terms.dot(device,rigid.rp.handle(),rigid.r.handle(),dim,&work.fold,&mut work.scalar)?;
            delta_energy=-0.5*(dr0+dr);
            if warm_start_policy::accept_energy(delta_energy,warm_verified) {
                combine(device,candidate,candidate,rigid.xr.handle(),1.0,0.0,dim)?;
                warm_accepted=true;
            }
        }
        if !warm_accepted {combine(device,rigid.z.handle(),rigid.z.handle(),rigid.r.handle(),1.0,0.0,dim)?;}
        eprintln!("PPF_PCG_WARM_START_RANK2 {} history={} coeff={:?} energy={:.9e} residual={:.9e} budget=2",
            if warm_accepted {"accepted"} else {"rejected"},usable_warm_count,coeff,delta_energy,warm_verified);
    } else if warm_eligible {
        let reason = if warm_has_history && warm_auto_cost && !auto_candidate {"auto_cost_cold"} else {match prior_warm_key.as_ref() {
            None => "no_history",
            Some(key) if key.dt_bits != warm_key.as_ref().unwrap().dt_bits => "dt_changed",
            Some(_) => "space_or_mask_changed",
        }};
        eprintln!("PPF_PCG_WARM_START cold reason={}", reason);
    }

    if arithmetic_active {
        reduction.prepare_arithmetic_status(device, record_arithmetic)?;
    }

    static SCALAR_DEBUG_SOLVES:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);
    let scalar_debug_selected=std::env::var("PPF_BODY_SCALAR64_SOLVE").ok().and_then(|v|v.parse::<usize>().ok());
    let group_debug_selected=std::env::var("PPF_BODY_GROUP_SOLVE").ok().and_then(|v|v.parse::<usize>().ok());
    if scalar_debug_selected.is_some() && group_debug_selected.is_some(){return Err(Fault::Platform{call:"body group diagnostic",detail:"scalar64 and group diagnostic are mutually exclusive".into()}.into());}
    let scalar_debug_id=if (scalar_debug_selected.is_some()||group_debug_selected.is_some()) && rigid.body_inverse.is_some(){Some(SCALAR_DEBUG_SOLVES.fetch_add(1,std::sync::atomic::Ordering::Relaxed)+1)}else{None};
    let group_debug=scalar_debug_id.is_some() && scalar_debug_id==group_debug_selected;
    let mut scalar_debug=if scalar_debug_id==scalar_debug_selected && scalar_debug_id.is_some(){
        let t=std::time::Instant::now();let x=rigid_failure_read_f32(device,rigid.xr.handle(),dim)?;let read_ms=t.elapsed().as_secs_f64()*1000.;let t=std::time::Instant::now();let norm=x.iter().map(|v|(*v as f64).powi(2)).sum::<f64>().sqrt();let sum_ms=t.elapsed().as_secs_f64()*1000.;
        eprintln!("PPF_BODY_SCALAR64_CONTEXT solve={} start={:.17e} sample={:.9e} dt={:.9e} tolerance={:.9e} err0={:.9e} groups={:?}",scalar_debug_id.unwrap(),joints.start_time_s,joints.sample_time_s,joints.dt,tolerance,err0,rigid.group_initial);
        Some(ScalarDebug{solve:scalar_debug_id.unwrap(),rz:0.,reads:1,bytes:dim*4,read_ms,sum_ms,x0norm:norm})
    }else{None};
    let mut factor_restarted=false;
    macro_rules! precondition_current {
        () => {{ precondition_current!(true); }};
        ($project_after:expr) => {{
            if let Some(ref body)=rigid.body_factor {
                device.run("pcg.rigid.precond_factor", |encoder| {
                    reduction.encode_precond_cloth_fp32(encoder,inverse_diagonal,rigid.r.handle(),rigid.z.handle())?;
                    body.encode_apply(encoder,rigid.r.handle(),rigid.z.handle(),body_base as u32,None)
                })?;
                match body.check_fault(device) {
                    Ok(())=>{},
                    Err(Fault::Platform{call:"body factor device",detail}) if detail=="device factor fault"=>{
                        let mut owner=rigid.body_factor.take().expect("factor owner");
                        owner.free(device)?;
                        device.copy(rigid.xr.handle(),0,rigid.body_seed_x.handle(),0,dim*4)?;
                        device.copy(rigid.r.handle(),0,rigid.body_seed_r.handle(),0,dim*4)?;
                        reduction.apply_precond(device,factor,inverse_diagonal,rigid.r.handle(),rigid.z.handle())?;
                        factor_restarted=true;
                        eprintln!("PPF_BODY_TREE_PCG fallback=cold_bj_restart reason=factor_fault");
                    },
                    Err(e)=>return Err(e.into()),
                }
            } else if arithmetic_active {
                arithmetic_applies += 1;
                device.run("pcg.rigid.precond_fp16_arithmetic", |encoder| {
                    reduction.encode_precond_selected(encoder, factor, inverse_diagonal,
                        rigid.r.handle(), rigid.z.handle(), Some(record_arithmetic))
                })?;
            } else {
                reduction.apply_precond(device, factor, inverse_diagonal,
                    rigid.r.handle(), rigid.z.handle())?;
            }
            if let Some(ref body)=rigid.body_inverse {
                let t=std::time::Instant::now();
                body.apply(device,rigid.r.span(body_base,6*bodies),rigid.z.span(body_base,6*bodies))?;
                if std::env::var_os("PPF_BODY_TREE_APPLY_TIMING").is_some_and(|v|v=="1"){eprintln!("PPF_BODY_TREE_APPLY elapsed_ms={:.6}",t.elapsed().as_secs_f64()*1000.);}
            }
            if let Some(ref mut host)=rigid.body_host_tree {
                host.apply(device,rigid.r.handle(),rigid.z.handle(),body_base,bodies)?;
            }
            // Packet callers project after this original fault observation.
            // The cold restart always retains the original projection.
            if $project_after || factor_restarted {
                project_tangent!(rigid.z.handle());
            }
        }};
    }

    macro_rules! seed_direction {
        () => {{
            precondition_current!();
            combine(device, rigid.z.handle(), rigid.p.handle(), rigid.p.handle(), 1.0, 0.0, dim)?;
            project_tangent!(rigid.p.handle());
            factor_restarted=false;
            if let Some(debug)=scalar_debug.as_mut(){debug.rz=debug.dot(device,rigid.r.handle(),rigid.z.handle(),dim)?;debug.rz as f32}else{terms.dot(device, rigid.r.handle(), rigid.z.handle(), dim, &work.fold, &mut work.scalar)?}
        }};
    }
    let mut rz: f32;

    // A recursive residual can drift away from f - A*x in finite precision.
    // Verify only claimed convergence, against the ORIGINAL projected seed's
    // group scales. Canonicalize x exactly as finish does, including root
    // particulars, then apply the affine operator WITHOUT projecting its input.
    // The verification must include prescribed affine rows in A*x.
    let true_residual_diagnostics = std::env::var_os("PPF_PCG_TRUE_RESIDUAL_DIAGNOSTICS")
        .is_some_and(|value| value == "1");
    let true_residual_capture_dir = std::env::var_os("PPF_PCG_TRUE_RESIDUAL_CAPTURE_DIR");
    let true_residual_capture_stop = std::env::var_os("PPF_PCG_TRUE_RESIDUAL_CAPTURE_STOP")
        .is_some_and(|value| value == "1");
    let mut true_residual_checks = if warm_accepted {1u32} else {0u32};
    let mut inertial_probe2_selected = false;
    let mut inertial_probe3_data: Option<(Vec<f32>,Vec<u32>,Vec<f32>)> = None;
    macro_rules! verify_true_residual {
        ($iteration:expr, $recursive:expr) => {{
            true_residual_checks += 1;
            let probe2_inputs = if inertial_probe2_selected || inertial_probe3_data.is_some() {
                Some((rigid_failure_read_f32(device,rigid.r.handle(),dim)?,rigid_failure_read_f32(device,rigid.xr.handle(),dim)?))
            } else { None };
            let capture_inputs = if true_residual_checks == 1 && true_residual_capture_dir.is_some() {
                Some((rigid_failure_read_f32(device,rigid.r.handle(),dim)?,
                      rigid_failure_read_f32(device,rigid.xr.handle(),dim)?))
            } else { None };
            let group_inputs=if group_debug {
                group_l1_device(device,rigid.r.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
                let recursive=rigid.group_current.clone();
                group_l1_device(device,rigid.f.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
                Some((recursive,rigid.group_current.clone()))
            }else{None};
            let diagnose = group_debug || (true_residual_diagnostics &&
                (true_residual_checks <= 4 || true_residual_checks.is_power_of_two()));
            // z is dead at claimed convergence and will be rebuilt on restart.
            // Reuse it only for bounded diagnostics of the finish projection;
            // no RHS, original scale, operator or returned iterate is changed.
            if diagnose {
                combine(device, rigid.xr.handle(), rigid.xr.handle(), rigid.z.handle(), 1.0, 0.0, dim)?;
            }
            project_tangent!(rigid.xr.handle());
            if input == RigidSolveInput::NewtonForce {
                if let Some(root)=roots { root.particular(device, joints.state, rigid.xr.handle())?; }
            }
            let canonicalization = if diagnose {
                combine(device, rigid.xr.handle(), rigid.z.handle(), rigid.z.handle(), 1.0, -1.0, dim)?;
                group_l1_device(device, rigid.z.handle(), &mut rigid.group_norm,
                    &work.fold, body_base, bodies, &mut rigid.group_current)?;
                Some(rigid.group_current.clone())
            } else { None };
            apply_reduced!(rigid.xr.handle(), rigid.rp.handle(), false);
            let product_norm = if diagnose {
                group_l1_device(device, rigid.rp.handle(), &mut rigid.group_norm,
                    &work.fold, body_base, bodies, &mut rigid.group_current)?;
                Some(rigid.group_current.clone())
            } else { None };
            measure_reference_residual!(group_current);
            let verified = super::pdrd::worst_relative_residual(
                &rigid.group_current, &rigid.group_initial, err0);
            if let Some((s0,free_rows,cold_r0))=inertial_probe3_data.as_ref() {
                let slots:Vec<usize>=free_rows.iter().enumerate().filter_map(|(i,&v)|if v!=0 {Some(i)} else {None}).collect();
                if dim==0 || dim>1024 || slots.is_empty() {return Err(Fault::Platform{call:"inertial probe3",detail:"unsupported diagnostic extent; intentional no-commit".into()}.into());}
                let recursive=probe2_inputs.as_ref().unwrap().0.clone();
                let current_x=rigid_failure_read_f32(device,rigid.xr.handle(),dim)?;
                let reference_r=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
                let rhs=rigid_failure_read_f32(device,rigid.f.handle(),dim)?;
                let (columns,a0)=inertial_probe_columns(device,op,reduction,joints,&rigid.zero)?;
                if columns.iter().chain(a0.iter()).chain(cold_r0.iter()).chain(rhs.iter()).chain(s0.iter()).any(|v|!v.is_finite()) {return Err(Fault::Platform{call:"inertial probe3",detail:"nonfinite model; intentional no-commit".into()}.into());}
                let n=slots.len();
                let a=nalgebra::DMatrix::<f64>::from_fn(n,n,|i,j|f64::from(columns[slots[j]*dim+slots[i]])-f64::from(a0[slots[i]]));
                let mut b=nalgebra::DVector::<f64>::zeros(n);
                for (i,&row) in slots.iter().enumerate(){b[i]=f64::from(cold_r0[row]);}
                let lu=a.clone().full_piv_lu();
                let delta=lu.solve(&b).ok_or_else(||Fault::Platform{call:"inertial probe3",detail:"singular model; intentional no-commit".into()})?;
                let model_residual=&b-&a*&delta;
                let model_error=model_residual.norm();
                let mut candidate=s0.clone();
                for (i,&slot) in slots.iter().enumerate(){candidate[slot]=(f64::from(s0[slot])+delta[i]) as f32;}
                if candidate.iter().any(|v|!v.is_finite()) {return Err(Fault::Platform{call:"inertial probe3",detail:"nonfinite rounded candidate; intentional no-commit".into()}.into());}
                let rounded_delta=nalgebra::DVector::<f64>::from_iterator(n,slots.iter().map(|&i|f64::from(candidate[i])-f64::from(s0[i])));
                let rounded_model_error=(&a*&rounded_delta-&b).norm();
                let mut evaluations=Vec::new();
                for pass in 0..2 {
                    rigid.z.write(device,0,&candidate)?;
                    apply_reduced!(rigid.z.handle(),rigid.rp.handle(),false);
                    combine(device,rigid.f.handle(),rigid.rp.handle(),rigid.r.handle(),1.,-1.,dim)?;
                    project_tangent!(rigid.r.handle());
                    group_l1_device(device,rigid.r.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
                    let actual=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
                    let score=super::pdrd::worst_relative_residual(&rigid.group_current,&rigid.group_initial,err0);
                    evaluations.push(serde_json::json!({"pass":pass,"candidate":candidate,"actual_gpu_r":actual,"actual_groups":rigid.group_current,"score":score}));
                    if pass==0 {let residual=nalgebra::DVector::<f64>::from_iterator(n,slots.iter().map(|&i|f64::from(actual[i])));
                        let correction=lu.solve(&residual).ok_or_else(||Fault::Platform{call:"inertial probe3",detail:"correction solve failed; intentional no-commit".into()})?;
                        for (i,&slot) in slots.iter().enumerate(){candidate[slot]=(f64::from(candidate[slot])+correction[i]) as f32;}
                        if candidate.iter().any(|v|!v.is_finite()){return Err(Fault::Platform{call:"inertial probe3",detail:"nonfinite correction; intentional no-commit".into()}.into());}
                    }
                }
                let faults=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;
                let doc=serde_json::json!({"kind":"inertial_probe3_fp32_basis_sampled_model","dim":dim,"body_base":body_base,"columns_column_major":columns,"a0":a0,"free_rows":free_rows,"cold_r0":cold_r0,"rhs":rhs,"cold_seed":s0,"current_x":current_x,"recursive_r":recursive,"reference_r":reference_r,"group_initial":rigid.group_initial,"err0":err0,"tolerance":tolerance,"delta_f64":delta.as_slice(),"model_f64_residual_free":model_residual.as_slice(),"model_f64_residual_l2":model_error,"rounded_model_residual_l2":rounded_model_error,"evaluations":evaluations,"faults":faults});
                let path=std::env::var("PPF_PCG_INERTIAL_PROBE_OUTPUT").map_err(|_|Fault::Platform{call:"inertial probe3",detail:"missing output path; intentional no-commit".into()})?;
                use std::io::Write;
                let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(&path).map_err(|e|Fault::Platform{call:"inertial probe3",detail:format!("open output: {e}")})?;
                file.write_all(&serde_json::to_vec_pretty(&doc).map_err(|e|Fault::Platform{call:"inertial probe3",detail:format!("serialize: {e}")})?).map_err(|e|Fault::Platform{call:"inertial probe3",detail:format!("write output: {e}")})?;
                eprintln!("PPF_INERTIAL_PROBE3 output={} dim={} free={} model_f64_residual_l2={:.12e} evaluations={}",path,dim,n,model_error,doc["evaluations"]);
                return Err(Fault::Platform{call:"inertial probe3",detail:"two candidate evaluations complete; intentional no-commit".into()}.into());
            }
            if let Some((recursive,pre_x))=probe2_inputs {
                let canonical_x=rigid_failure_read_f32(device,rigid.xr.handle(),dim)?;
                let r_ref=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
                let ref_groups=rigid.group_current.clone();
                apply_reduced!(rigid.xr.handle(),rigid.rp.handle(),true);
                combine(device,rigid.f.handle(),rigid.rp.handle(),rigid.r.handle(),1.,-1.,dim)?;
                project_tangent!(rigid.r.handle());
                group_l1_device(device,rigid.r.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
                let krylov_groups=rigid.group_current.clone();
                let r_krylov=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
                let groups=|v:&[f32]| -> Vec<f64> {
                    let mut g=vec![0.0f64;bodies+1];
                    for (i,&x) in v.iter().enumerate() {let k=if i<body_base {0} else {1+(i-body_base)/6}; if k<g.len(){g[k]+=f64::from(x.abs());}} g
                };
                let difference=|a:&[f32],b:&[f32]| -> Vec<f32> {a.iter().zip(b).map(|(&x,&y)|x-y).collect()};
                let faults=super::shadow_tree::fault_snapshot(device,op,joints).map_err(super::scene::Fatal::invariant)?;
                eprintln!("PPF_INERTIAL_PROBE2 iteration={} cold_groups={:?} err0={:.9e} recursive_groups={:?} ref_groups={:?} krylov_groups={:?} ref_minus_krylov_groups={:?} recursive_minus_ref_groups={:?} xr_canonical_delta_groups={:?} roots={} lock={} body_projection={} grouped_soft={} coalesce={} fused={} arithmetic={} frozen_ball={} frozen_angular={} joint_count={} control_count={} extension_count={} faults={:?}",$iteration,rigid.group_initial,err0,groups(&recursive),ref_groups,krylov_groups,groups(&difference(&r_ref,&r_krylov)),groups(&difference(&recursive,&r_ref)),groups(&difference(&canonical_x,&pre_x)),roots.is_some(),lock.is_some(),reduction.map.needs_projection(),grouped_soft.is_some(),coalesce_mixed,fused_joint_product,arithmetic_active,joints.frozen_ball.is_some(),joints.frozen_angular.is_some(),joints.count,joints.control_count,joints.extension_count,faults);
                return Err(Fault::Platform{call:"inertial probe2",detail:"first convergence claim observed; intentional no-commit stop".into()}.into());
            }
            if diagnose {
                eprintln!("PCG_TRUE_RESIDUAL_VERIFY check={} iteration={} recursive={:.9e} true={:.9e} tolerance={:.9e} err0={:.9e} grouped_soft={} dim={} body_base={} bodies={} initial={:?} current={:?} affine_product_l1={:?} finish_delta_l1={:?}",
                    true_residual_checks,$iteration,$recursive,verified,tolerance,err0,
                    grouped_soft.is_some(),dim,body_base,bodies,rigid.group_initial,
                    rigid.group_current,product_norm,canonicalization);
            }
            if let Some((recursive,pre_finish_x)) = capture_inputs {
                let directory = std::path::Path::new(true_residual_capture_dir.as_ref().unwrap());
                true_residual_capture::capture(device,op,reduction,joints,roots,
                    lock.is_none(),rigid.xr.handle(),rigid.f.handle(),rigid.r.handle(),
                    &recursive,&pre_finish_x,&rigid.group_initial,err0,$iteration,directory,
                    serde_json::json!({"xr":format!("{:?}",rigid.xr.handle()),"rhs":format!("{:?}",rigid.f.handle()),
                        "r":format!("{:?}",rigid.r.handle()),"z":format!("{:?}",rigid.z.handle()),
                        "p":format!("{:?}",rigid.p.handle()),"rp":format!("{:?}",rigid.rp.handle()),
                        "xv":format!("{:?}",rigid.xv.handle()),"mxv":format!("{:?}",rigid.mxv.handle()),
                        "fold":format!("{:?}",work.fold.handle()),"row_absolute":format!("{:?}",work.row_absolute.handle()),
                        "row_curvature":format!("{:?}",work.row_curvature.handle())}))?;
                if true_residual_capture_stop {
                    return Err(Fault::Platform {call:"true residual diagnostic stop",
                        detail:"first convergence claim captured; diagnostic refusal, not a physical result".to_owned()}.into());
                }
            }
            if let Some((recursive_groups,rhs_groups))=group_inputs {
                let first=rigid.group_current.clone();
                if let Some(reference)=rigid.body_group_reference.as_ref(){
                    body_group_reference(device,reference,rigid.xr.handle(),rigid.f.handle(),rigid.rp.handle(),rigid.r.handle(),body_base,bodies,&rigid.group_initial,err0,joints)?;
                }else{eprintln!("PPF_BODY_GROUP_REFERENCE unsupported=missing_direct");}
let mut product=Buffer::<f32>::default();let mut residual=Buffer::<f32>::default();
                let comparison=(|| -> super::scene::FatalResult<(Vec<f32>,Vec<f32>)> {
                    product.size(device,dim,ppf_cts_compute::AllocLabel("body_group.product"))?;
                    residual.size(device,dim,ppf_cts_compute::AllocLabel("body_group.residual"))?;
                    apply_reduced!(rigid.xr.handle(),product.handle(),false);
                    combine(device,rigid.f.handle(),product.handle(),residual.handle(),1.0,-1.0,dim)?;
                    project_tangent!(residual.handle());
                    group_l1_device(device,residual.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
                    let second=rigid.group_current.clone();
                    combine(device,residual.handle(),rigid.r.handle(),product.handle(),1.0,-1.0,dim)?;
                    group_l1_device(device,product.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
                    Ok((second,rigid.group_current.clone()))
                })();
                let free_product=product.free(device);let free_residual=residual.free(device);
                let (second,repeat_delta)=match comparison {Err(e)=>return Err(e),Ok(v)=>{free_product?;free_residual?;v}};
                let mut worst=0usize;let mut worst_ratio=0f32;
                for i in 0..first.len(){let scale=if rigid.group_initial[i]>0.{rigid.group_initial[i]}else{err0};let ratio=first[i]/scale;if !ratio.is_finite(){return Err(super::scene::Fatal::invariant("group diagnostic nonfinite"));}if ratio>worst_ratio{worst=i;worst_ratio=ratio;}}
                let diff:Vec<f64>=first.iter().zip(&second).map(|(&a,&b)|b as f64-a as f64).collect();
                eprintln!("PPF_BODY_GROUP_COMPARE solve={} iteration={} worst={} worst_ratio={:.9e} worst_initial={:.9e} worst_true={:.9e} initial={:?} recursive={:?} rhs={:?} product={:?} canonical_delta={:?} true_first={:?} true_second={:?} second_minus_first={:?} repeat_delta_l1={:?}",scalar_debug_id.unwrap(),$iteration,worst,worst_ratio,rigid.group_initial[worst],first[worst],rigid.group_initial,recursive_groups,rhs_groups,product_norm,canonicalization,first,second,diff,repeat_delta);
                return Err(Fault::Platform{call:"body group diagnostic",detail:"first true check; intentional no-commit".into()}.into());
            }
            if let Some(debug)=scalar_debug.as_ref(){debug.receipt($iteration,$recursive,verified,"truecheck");return Err(Fault::Platform{call:"scalar64 diagnostic",detail:"first true check; intentional no-commit".into()}.into());}
            verified
        }};
    }

    // A failed true-residual check starts a new Krylov direction, NOT a new
    // solve. M stays fixed except the opt-in one-time BJ fallback below;
    // never replace group_initial/err0 or replenish the cap.
    let mut no_progress = no_progress::RestartTracker::default();
    macro_rules! restart_from_true_residual {
        () => {{
            if let Some(debug)=rigid.body_inverse_debug.take() {
                let owner=rigid.body_inverse.as_ref().expect("debug owner");
                debug_body_apply(device,owner,&debug,reduction,factor,inverse_diagonal,roots,rigid.r.handle(),rigid.r.span(body_base,6*bodies),body_base,bodies)?;
                return Err(Fault::Platform{call:"body tree debug",detail:"first true residual restart; intentional no-commit".into()}.into());
            }
            let had_host=if let Some(host)=rigid.body_host_tree.take(){host.receipt();true}else{false};
            let had_device=if let Some(mut owner)=rigid.body_inverse.take(){owner.free(device)?;true}else{false};
            let had_factor=if let Some(mut owner)=rigid.body_factor.take(){owner.free(device)?;true}else{false};
            if had_host || had_device || had_factor {
                no_progress.clear();
                device.copy(rigid.xr.handle(),0,rigid.body_seed_x.handle(),0,dim*4)?;
                device.copy(rigid.r.handle(),0,rigid.body_seed_r.handle(),0,dim*4)?;
                eprintln!("PPF_BODY_TREE_PCG fallback={} reason=rejected_true_check", if lazy_tree_mode.is_some() { "activation_bj_restart" } else { "cold_bj_restart" });
            }
            rz = seed_direction!();
        }};
    }
    let mut completed_iterations = 0u32;

    // `x = P u` and the rotation export, which every exit below takes.
    macro_rules! finish {
        ($outcome:expr, $iteration:expr, $residual:expr) => {{
            if std::env::var_os("PPF_LEGACY_ROWS_SELF_CHECK").as_deref()==Some(std::ffi::OsStr::new("1")) {return Err(super::scene::Fatal::invariant("legacy rows self check not reached; no commit"));}
            if std::env::var_os("PPF_FACTOR_BATCH8_PACKET_SELF_CHECK").as_deref()==Some(std::ffi::OsStr::new("1")) {return Err(super::scene::Fatal::invariant("factor batch8 diagnostic not reached; no commit"));}
            if inertial_probe2_selected || inertial_probe3_data.is_some() { return Err(Fault::Platform{call:"inertial probe2",detail:"selected solve ended before comparison; intentional no-commit stop".into()}.into()); }
            // Converged exits already canonicalized this exact iterate before
            // evaluating its true residual. Do not project it a second time:
            // floating-point projectors need not be bitwise idempotent. Other
            // exits retain their original feasibility projection.
            if !matches!($outcome, Outcome::Converged | Outcome::AcceptedAnimation) {
                project_tangent!(rigid.xr.handle());
            }
            if input == RigidSolveInput::NewtonForce {
                if !matches!($outcome, Outcome::Converged | Outcome::AcceptedAnimation) {
                    if let Some(root)=roots { root.particular(device, joints.state, rigid.xr.handle())?; }
                }
            }
            export_rigid_solve_iterate(device, reduction, rigid.xr.handle(), x, rotation_out, input)?;
            if warm_eligible && matches!($outcome, Outcome::Converged | Outcome::AcceptedAnimation) && true_residual_checks > 0 {
                if requested_mode == 2 && usable_warm_count > 0 {
                    std::mem::swap(&mut rigid.warm_history,&mut rigid.warm_history2);
                }
                combine(device, rigid.xr.handle(), rigid.warm_seed.handle(), rigid.warm_history.handle(), 1.0, -1.0, dim)?;
                // Retain only the homogeneous correction on free reduced rows.
                // Prescribed rows stay in the current affine seed, so a later
                // candidate can never carry an older pin target forward.
                if warm_key.as_ref().is_some_and(|key| key.mask.iter().any(|&value| value != 0)) {
                    warm_mask_history(device, rigid.warm_history.handle(), rigid.warm_free_rows.handle(), dim)?;
                }
                rigid.warm_key = warm_key.take();
                rigid.warm_count = if requested_mode == 2 {(usable_warm_count + 1).min(2)} else {1};
                if warm_auto_cost {
                    cold_cost.record_success($iteration,true,warm_accepted);
                    rigid.warm_cold_cost = cold_cost;
                }
            }
            // THE REDUCED SOLVE'S OWN REPORT. It
            // names the WORST DOF GROUP's relative residual rather than a norm
            // over the whole reduced vector, which is the whole point of the
            // per-group measure: one heavy body's exactly-preconditioned block
            // is annihilated in a single step, so a global norm crosses the
            // tolerance while the cloth still carries its full residual.
            ::log::info!(
                "reduced PDRD solve took {} iteration(s), worst DOF group at a \
                 relative residual of {:.9}{}",
                $iteration,
                $residual,
                if matches!($outcome, Outcome::CurvatureTruncated) {
                    " (curvature truncated)"
                } else {
                    ""
                }
            );
            if let Some((started, before)) = timing_receipt {
                let after = device.counters();
                eprintln!("PPF_PCG_TIMING_RECEIPT input={} coalesce_eligible={} coalesced_iterations={} grouped_soft={} bodies={} body_projection={} roots={} lock={} schwarz={} outcome={:?} iterations={} true_residual_checks={} residual={:.9e} elapsed_ms={:.3} syncs={} dispatches={} readback_requests=unavailable readback_bytes=unavailable",
                    if input == RigidSolveInput::NewtonForce {"newton"} else {"reduced_rhs"},
                    coalesce_mixed,coalesced_iterations,grouped_soft.is_some(),bodies,
                    reduction.map.needs_projection(),roots.is_some(),lock.is_some(),
                    reduction.staged.reduced_schwarz.active(),$outcome,$iteration,
                    true_residual_checks,$residual,started.elapsed().as_secs_f64()*1000.0,
                    after.syncs.saturating_sub(before.syncs),after.dispatches.saturating_sub(before.dispatches));
            }
            if observe_arithmetic {
                let (half_rows, fallback_rows) = if record_arithmetic {
                    reduction.observe_arithmetic_status(device)?
                } else {
                    (0, 0)
                };
                eprintln!("PPF_PCG_ARITHMETIC_RECEIPT eligible={} applies={} used_fp16_rows={} fallback_fp32_rows={} downgrade={}",
                    arithmetic_eligible, arithmetic_applies, half_rows, fallback_rows,
                    arithmetic_downgrade.unwrap_or("none"));
            }
            return Ok(Report {
                outcome: $outcome,
                iterations: $iteration,
                relative_residual: $residual,
            });
        }};
    }

    if warm_accepted && warm_verified < tolerance {
        finish!(Outcome::Converged, 0, warm_verified);
    }
    if std::env::var_os("PPF_PCG_INERTIAL_SEED").is_some_and(|v| v == "1") {
        let eligible = inertial_context.is_some() && warm_mask.is_some()
            && input == RigidSolveInput::NewtonForce && resident_without_hard
            && reduction.staged.pcg_precision == crate::args::PcgPrecision::Fp32
            && requested_mode == 0 && grouped_soft.is_none() && !coalesce_mixed
            && !fused_joint_product && !arithmetic_active
            && !reduction.staged.reduced_schwarz.active()
            && rigid.body_inverse.is_none() && rigid.body_host_tree.is_none() && rigid.body_factor.is_none()
            && scalar_debug.is_none() && !group_debug && failure_capture_dir.is_none()
            && true_residual_capture_dir.is_none() && !true_residual_diagnostics
            && !true_residual_capture_stop && lanczos_solve_id.is_none()
            && !std::env::var_os("PPF_JOINT_RESIDENT_PCG").is_some_and(|v| v == "1");
        if !eligible {
            eprintln!("PPF_PCG_INERTIAL_SEED fallback reason=incompatible_mode");
        } else {
            let started=std::time::Instant::now();
            static PROBE_SOLVES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let probe_mode = std::env::var("PPF_PCG_INERTIAL_PROBE").unwrap_or_default();
            let probe = if probe_mode=="1" || probe_mode=="2" || probe_mode=="3" {
                PROBE_SOLVES.fetch_add(1,std::sync::atomic::Ordering::Relaxed)+1
            } else { 0 };
            if (probe_mode=="2" || probe_mode=="3") && probe==2 {
                if roots.is_some() || lock.is_some() || reduction.map.needs_projection() || !resident_without_hard {
                    return Err(Fault::Platform{call:"inertial probe2",detail:"unsupported nonidentity domain; intentional no-commit stop".into()}.into());
                }
                if probe_mode=="2" { inertial_probe2_selected=true; }
            }
            let ctx=inertial_context.unwrap();
            let n=reduction.map.nrow;
            let read_start=std::time::Instant::now();
            let eval=rigid_failure_read_f32(device,ctx.eval_x,3*n)?;
            let target=rigid_failure_read_f32(device,ctx.target,3*n)?;
            let prot=rigid_failure_read_f32(device,reduction.staged.prot.handle(),3*n)?;
            let mask=rigid_failure_read_u32(device,warm_mask.unwrap(),n)?;
            let cold_x=rigid_failure_read_f32(device,rigid.xr.handle(),dim)?;
            let cold_r=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
            let cold_rp=rigid_failure_read_f32(device,rigid.rp.handle(),dim)?;
            let mut props=vec![crate::data::PdrdBodyProp::default();bodies];
            if bodies>0 {
                let bytes=std::slice::from_raw_parts_mut(props.as_mut_ptr().cast::<u8>(), bodies*std::mem::size_of::<crate::data::PdrdBodyProp>());
                device.read(reduction.staged.body_prop.handle(),0,bytes)?;
            }
            let read_ms=read_start.elapsed().as_secs_f64()*1000.;
            if (1..=2).contains(&probe) {
                let l1=|v:&[f32]| v.iter().map(|x|f64::from(x.abs())).sum::<f64>();
                eprintln!("PPF_INERTIAL_PROBE cold solve={} err0={:.9e} groups={:?} xr_l1={:.12e} r_l1={:.12e} rp_l1={:.12e} roots={} lock={} body_projection={} grouped_soft={} coalesce={} fused={} arithmetic={} resident_requested={}",probe,err0,rigid.group_initial,l1(&cold_x),l1(&cold_r),l1(&cold_rp),roots.is_some(),lock.is_some(),reduction.map.needs_projection(),grouped_soft.is_some(),coalesce_mixed,fused_joint_product,arithmetic_active,std::env::var_os("PPF_JOINT_RESIDENT_PCG").is_some_and(|v|v=="1"));
            }
            let fit_start=std::time::Instant::now();
            let mass:Vec<f32>=props.iter().map(|p|p.mass_per_vertex).collect();
            let fit=super::pcg_inertial_seed::fit(reduction.map,&prot,&eval,&target,&mass);
            let free=warm_start_policy::reduced_free_rows(reduction.map,&mask);
            if probe_mode=="3" && probe==2 { inertial_probe3_data=Some((cold_x.clone(),free.clone().ok_or_else(||Fault::Platform{call:"inertial probe3",detail:"invalid free mask; intentional no-commit".into()})?,cold_r.clone())); }
            let fit_ms=fit_start.elapsed().as_secs_f64()*1000.;
            if let (Some(q),Some(free))=(fit,free) {
                let delta:Vec<f32>=q.iter().zip(&cold_x).zip(&free).map(|((&a,&b),&f)|if f==0 {0.} else {a-b}).collect();
                if delta.iter().all(|v|v.is_finite()) {
                    let write_start=std::time::Instant::now();
                    rigid.p.write(device,0,&delta)?;
                    project_tangent!(rigid.p.handle());
                    combine(device,rigid.xr.handle(),rigid.p.handle(),rigid.z.handle(),1.,1.,dim)?;
                    let write_ms=write_start.elapsed().as_secs_f64()*1000.;
                    let validate_start=std::time::Instant::now();
                    apply_reduced!(rigid.z.handle(),rigid.rp.handle(),false);
                    combine(device,rigid.f.handle(),rigid.rp.handle(),rigid.r.handle(),1.,-1.,dim)?;
                    project_tangent!(rigid.r.handle());
                    group_l1_device(device,rigid.r.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
                    let score=super::pdrd::worst_relative_residual(&rigid.group_current,&rigid.group_initial,err0);
                    let cold=super::pdrd::worst_relative_residual(&rigid.group_initial,&rigid.group_initial,err0);
                    let accepted=score.is_finite() && score<cold;
                    if (1..=2).contains(&probe) { eprintln!("PPF_INERTIAL_PROBE candidate solve={} score={:.9e} groups={:?} accepted={}",probe,score,rigid.group_current,accepted); }
                    if accepted { combine(device,rigid.z.handle(),rigid.z.handle(),rigid.xr.handle(),1.,0.,dim)?; }
                    else { rigid.r.seed(device,&cold_r)?; rigid.rp.write(device,0,&cold_rp)?; }
                    if (1..=2).contains(&probe) && !accepted {
                        let restored_r=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
                        let restored_rp=rigid_failure_read_f32(device,rigid.rp.handle(),dim)?;
                        let maxdiff=|a:&[f32],b:&[f32]| a.iter().zip(b).map(|(&x,&y)|(x-y).abs()).fold(0.0f32,f32::max);
                        let restore_r=maxdiff(&restored_r,&cold_r);
                        let restore_rp=maxdiff(&restored_rp,&cold_rp);
                        apply_reduced!(rigid.xr.handle(),rigid.rp.handle(),false);
                        combine(device,rigid.f.handle(),rigid.rp.handle(),rigid.r.handle(),1.,-1.,dim)?;
                        project_tangent!(rigid.r.handle());
                        group_l1_device(device,rigid.r.handle(),&mut rigid.group_norm,&work.fold,body_base,bodies,&mut rigid.group_current)?;
                        let recomputed=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
                        eprintln!("PPF_INERTIAL_PROBE reject solve={} restore_r_max={:.12e} restore_rp_max={:.12e} recompute_r_max={:.12e} recompute_groups={:?}",probe,restore_r,restore_rp,maxdiff(&recomputed,&cold_r),rigid.group_current);
                        rigid.r.seed(device,&cold_r)?;
                        rigid.rp.write(device,0,&cold_rp)?;
                    }
                    if probe_mode=="1" && probe==2 { return Err(Fault::Platform{call:"inertial probe",detail:"two solves observed; intentional no-commit stop".into()}.into()); }
                    let validation_ms=validate_start.elapsed().as_secs_f64()*1000.;
                    eprintln!("PPF_PCG_INERTIAL_SEED {} reason={} score={:.9e} cold={:.9e} read_ms={:.3} fit_ms={:.3} write_ms={:.3} validation_ms={:.3} total_ms={:.3}",if accepted {"accepted"} else {"rejected"},if accepted {"improved"} else {"no_improvement"},score,cold,read_ms,fit_ms,write_ms,validation_ms,started.elapsed().as_secs_f64()*1000.);
                    if accepted && score < tolerance {
                        let verified = verify_true_residual!(0, score);
                        if verified < tolerance { finish!(Outcome::Converged, 0, verified); }
                    }
                } else { eprintln!("PPF_PCG_INERTIAL_SEED fallback reason=nonfinite_delta read_ms={:.3} fit_ms={:.3} total_ms={:.3}",read_ms,fit_ms,started.elapsed().as_secs_f64()*1000.); }
            } else { eprintln!("PPF_PCG_INERTIAL_SEED fallback reason=fit_or_mask read_ms={:.3} fit_ms={:.3} total_ms={:.3}",read_ms,fit_ms,started.elapsed().as_secs_f64()*1000.); }
            if probe_mode=="1" && probe==2 { return Err(Fault::Platform{call:"inertial probe",detail:"two solves observed; intentional no-commit stop".into()}.into()); }
        }
    }
    let animation_start=std::time::Instant::now();
    let factor_origin=rigid.body_factor.is_some();
    let animation_request=std::env::var("PPF_PCG_ANIMATION_MOTION_TOL").ok().and_then(|s|s.parse::<f64>().ok()).filter(|x|x.is_finite() && *x>0.0);
    let animation_eligible=animation_request.is_some() && inertial_context.is_some() && warm_mask.is_some()
        && input==RigidSolveInput::NewtonForce && resident_without_hard && lock.is_none() && !reduction.map.needs_projection()
        && reduction.staged.pcg_precision==crate::args::PcgPrecision::Fp32 && requested_mode==0 && grouped_soft.is_none()
        && !coalesce_mixed && !fused_joint_product && !arithmetic_active && !reduction.staged.reduced_schwarz.active()
        && rigid.body_inverse.is_none() && scalar_debug.is_none() && !group_debug
        && failure_capture_dir.is_none() && true_residual_capture_dir.is_none() && !true_residual_diagnostics
        && !true_residual_capture_stop && lanczos_solve_id.is_none()
        && (!std::env::var_os("PPF_JOINT_RESIDENT_PCG").is_some_and(|v|v=="1") || factor_origin)
        && !std::env::var_os("PPF_PCG_INERTIAL_PROBE").is_some_and(|v|v=="1" || v=="2" || v=="3");
    let mut animation_model: Option<super::pcg_motion_stop::MotionModel> = None;
    let mut animation_model_attempted = false;
    macro_rules! animation_decision {
        () => {{
            let animation_mode_allowed=rigid.body_inverse.is_none() &&
                (!std::env::var_os("PPF_JOINT_RESIDENT_PCG").is_some_and(|v|v=="1") || factor_origin);
            if animation_eligible && animation_mode_allowed && !animation_model_attempted {
                animation_model_attempted = true;
                let model_started = std::time::Instant::now();
                let actual_preconditioner = if rigid.body_factor.is_some() { "factor" } else if rigid.body_host_tree.is_some() { "host_tree" } else { "bj" };
                animation_model = animation_motion_model(device,reduction,inertial_context.unwrap(),warm_mask.unwrap(),roots,joints.dt)?;
                if animation_model.is_none() { eprintln!("PPF_PCG_ANIMATION fallback reason=model_unavailable actual_preconditioner={}",actual_preconditioner); }
                else { eprintln!("PPF_PCG_ANIMATION model_builder_ms={:.3} delta={:.9e} metric=inertial_motion_estimate actual_preconditioner={}",model_started.elapsed().as_secs_f64()*1000.,animation_request.unwrap(),actual_preconditioner); }
            }
            if let Some(model)=animation_model.as_ref().filter(|_| animation_mode_allowed) {
                let relative:Vec<bool>=rigid.group_current.iter().zip(&rigid.group_initial).map(|(&now,&start)|{
                    let scale=if start>0.0 {start}else{err0}; let ratio=now/scale; ratio.is_finite() && ratio<tolerance
                }).collect();
                let values=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
                let metrics=model.evaluate(&values);
                let decision=super::pcg_motion_stop::decide(&relative,&metrics,animation_request.unwrap());
                if let Some(ref d)=decision {if d.accepted && !d.motion_groups.is_empty() {
                    eprintln!("PPF_PCG_ANIMATION accepted reason=AnimationMotion delta={:.9e} motion_groups={:?} max_motion={:?} cold_groups={:?} actual_groups={:?} elapsed_since_request_ms={:.3} actual_preconditioner={}",animation_request.unwrap(),d.motion_groups,d.max_failed_relative_motion,rigid.group_initial,rigid.group_current,animation_start.elapsed().as_secs_f64()*1000.,if rigid.body_factor.is_some() { "factor" } else if rigid.body_host_tree.is_some() { "host_tree" } else { "bj" });
                    true
                }else{false}}else{false}
            }else{false}
        }};
    }
    rz = seed_direction!();
    factor_restarted=false;
    if arithmetic_active && (!rz.is_finite() || rz <= 0.0) {
        arithmetic_active = false;
        arithmetic_downgrade = Some("initial_preconditioned_residual");
        rz = seed_direction!();
    }

    // The arithmetic preconditioner may vary as the residual is rounded.
    // Switch back once per solve. Recompute the exact residual and discard
    // the direction, but keep the original norm scales and iteration cap.
    macro_rules! downgrade_and_restart {
        ($reason:expr, $iteration:expr, $recursive:expr) => {{
            no_progress.clear();
            arithmetic_active = false;
            arithmetic_downgrade = Some($reason);
            let verified = verify_true_residual!($iteration, $recursive);
            if verified < tolerance {
                finish!(Outcome::Converged, $iteration, verified);
            }
            if $iteration >= max_iterations || !verified.is_finite() {
                finish!(Outcome::MaxIterations, $iteration, verified);
            }
            restart_from_true_residual!();
        }};
    }

    // Experimental mixed-joint resident stride one. A fresh device stop per
    // solve preserves group-relative scales; speculative z is auxiliary on stop.
    // Root projection in this path is opt-in until its A/B tests complete.
    let resident_root = roots.is_some()
        && (auto_resident_active || std::env::var_os("PPF_JOINT_RESIDENT_ROOT").is_some_and(|x| x == "1"));
    let resident_joint = !fused_joint_product && (auto_resident_active || std::env::var_os("PPF_JOINT_RESIDENT_PCG").is_some_and(|x| x == "1"))
        && input == RigidSolveInput::NewtonForce
        && resident_without_hard
        && bodies > 0
        && !arithmetic_active
        && joints.count != 0
        && joints.can_encode_krylov()
        && reduction.staged.pcg_precision == crate::args::PcgPrecision::Fp32
        && !reduction.map.needs_projection()
        && (roots.is_none() || resident_root)
        && lock.is_none()
        && !reduction.staged.reduced_schwarz.active()
        && failure_capture_dir.is_none()
        && true_residual_capture_dir.is_none();
    let mut factor_refresh_pending: Option<(u32,f32)> = None;
    let factor_iteration_started=std::time::Instant::now();
    if resident_joint {
        // Purely host-side: only the already-downloaded scalar slots are read.
        // A different route, flexible preconditioner, warm seed, or batch
        // replay cannot contribute a segment to this fixed-M recurrence.
        let mut lanczos = if requested_mode == 0
            && !std::env::var_os("PPF_JOINT_RESIDENT_BATCH8").is_some_and(|x|x=="1") {
            lanczos_solve_id.and_then(|id|lanczos_trace::Trace::new(id,joints.start_time_s,
                joints.sample_time_s,joints.dt,dim,body_base,bodies))
        } else {None};
        // Every soft family must have a current frozen cache; rigid frozen
        // readiness is already established by can_encode_krylov above.
        // Root's projection has no stop latch. Until a guarded kernel exists,
        // observe every root iteration on the host; never replay it in batch8.
        let batch_admitted = roots.is_none()
            && std::env::var_os("PPF_JOINT_RESIDENT_BATCH8").is_some_and(|x| x == "1")
            && joints.soft.map_or(true, |soft| soft.batch8_ready())
            && max_iterations >= 16;
        let prefix_end = if batch_admitted { 8 } else { max_iterations };
        let mut factor_refresh_trial = if auto_resident_active && !batch_admitted
            && rigid.active_factor_borrowed && rigid.body_factor.is_some()
        {
            factor_refresh_policy::Trial::configured()
        } else {
            None
        };
        let mut prefix_continues = true;
        ::log::info!("pcg.joint.resident_stride1 active roots={}", roots.is_some());
        if resident_root { eprintln!("PPF_JOINT_RESIDENT_ROOT active=true batch8=false"); }
        rigid
            .group_initial_device
            .write(device, 0, &rigid.group_initial)?;
        let mut seed = [0.0; slot::COUNT];
        seed[slot::RZ0] = rz;
        work.scalar.seed(device, &seed)?;
        if std::env::var_os("PPF_LEGACY_ROWS_SELF_CHECK").as_deref()==Some(std::ffi::OsStr::new("1")) {
            if let Some(soft)=joints.soft {
                if soft.legacy_rows_self_check(device,rigid.p.handle())? {
                    return Err(super::scene::Fatal::invariant("legacy rows self check complete; no commit"));
                }
            }
            return Err(super::scene::Fatal::invariant("legacy rows self check not reached; no commit"));
        }
        if std::env::var_os("PPF_FACTOR_BATCH8_PACKET_SELF_CHECK").as_deref()==Some(std::ffi::OsStr::new("1")) {
            if let Some(ref owner)=rigid.body_factor {
                super::body_factor_device::batch8_packet_self_check(device,owner)?;
                eprintln!("PPF_FACTOR_BATCH8_PACKET_SELF_CHECK passed; no commit");
                return Err(super::scene::Fatal::invariant("factor batch8 packet self check complete; no commit"));
            }
            return Err(super::scene::Fatal::invariant("factor batch8 packet self check requires factor; no commit"));
        }
        if std::env::var_os("PPF_BODY_FACTOR_PACKET_SELF_CHECK").as_deref()==Some(std::ffi::OsStr::new("1")) {
            if let Some(ref owner)=rigid.body_factor {
                super::body_factor_device::packet_self_check(device,owner)?;
                eprintln!("PPF_BODY_FACTOR_PACKET_SELF_CHECK passed; no commit");
                return Err(super::scene::Fatal::invariant("body factor packet self check complete; no commit"));
            }
        }
        let stride_graph_enabled = auto_resident_active && !batch_admitted
            && lanczos_solve_id.is_none() && device.info().supports_deferred_regions
            && std::env::var_os("PPF_JOINT_STRIDE_GRAPH").is_some_and(|value| value == "1");
        let stride_graph_receipt = stride_graph_enabled
            && std::env::var_os("PPF_JOINT_STRIDE_GRAPH_RECEIPT").is_some_and(|value| value == "1");
        let mut stride_graph = None::<D::Region>;
        let mut stride_graph_replays = 0u32;
        macro_rules! release_stride_graph {
            () => {
                if let Some(graph) = stride_graph.take() {
                    device.release(graph);
                    if stride_graph_receipt {
                        eprintln!("PPF_JOINT_STRIDE_GRAPH released=true replays={stride_graph_replays} roots={}", roots.is_some());
                    }
                }
            };
        }
        for iteration in 1..=prefix_end {
            if iteration == max_iterations {
                release_stride_graph!();
            }
            let direction = rigid.p.handle();
            let applied = rigid.rp.handle();
            let residual_h = rigid.r.handle();
            let preconditioned = rigid.z.handle();
            let rz_previous = work.scalar.span(slot::RZ0, 1);
            let pap = work.scalar.span(slot::P_AP, 1);
            let alpha = work.scalar.span(slot::ALPHA, 1);
            let rz_next = work.scalar.span(slot::RZ1, 1);
            let beta = work.scalar.span(slot::BETA, 1);
            let relative = work.scalar.span(slot::ALPHA_NOISE, 1);
            let status = work.scalar.span(slot::BETA_NOISE, 1);
            let factor_fault = work.scalar.span(slot::FACTOR_FAULT, 1);
            let products = terms.product_handle();
            let mut encode_stride = |encoder: &mut dyn Encoder| unsafe {
                // The affine iterate carries prescribed root values. Only the
                // homogeneous direction and output are projected here.
                if let Some(root) = roots { root.encode_project(encoder, direction)?; }
                reduction.encode_prolong(encoder, direction, rigid.xv.handle())?;
                if tet_force_only {
                    op.encode_apply_force_only(encoder, rigid.xv.handle(), rigid.mxv.handle(),
                        work.row_absolute.handle(), work.row_curvature.handle())?;
                } else {
                    op.encode_apply(encoder, rigid.xv.handle(), rigid.mxv.handle(),
                        work.row_absolute.handle(), work.row_curvature.handle())?;
                }
                reduction.encode_restrict(encoder, rigid.mxv.handle(), applied, allow_gather)?;
                joints.encode_krylov(encoder, direction, applied, row_parallel)?;
                if let Some(root) = roots { root.encode_project(encoder, applied)?; }
                terms.encode_fill(encoder, direction, applied, dim)?;
                encode_fold(encoder, products, terms.rows(), &work.fold, pap, false)?;
                encoder.elements(
                    &super::kernels::FemPcgReducedBatchAlphaArgs {
                        rz: rz_previous,
                        pap,
                        alpha,
                        status,
                        extent: 1,
                        count: 1,
                        seam_arena_count: 0,
                    },
                    1,
                )?;
                encoder.elements(
                    &super::kernels::FemPcgReducedBatchAddIndirectArgs {
                        source: direction,
                        destination: rigid.xr.handle(),
                        coefficient: alpha,
                        sign: 1.0,
                        status,
                        extent: dim as u32,
                        count: dim as u32,
                        seam_arena_count: 0,
                    },
                    dim as u32,
                )?;
                encoder.elements(
                    &super::kernels::FemPcgReducedBatchAddIndirectArgs {
                        source: applied,
                        destination: residual_h,
                        coefficient: alpha,
                        sign: -1.0,
                        status,
                        extent: dim as u32,
                        count: dim as u32,
                        seam_arena_count: 0,
                    },
                    dim as u32,
                )?;
                if let Some(root) = roots { root.encode_project(encoder, residual_h)?; }
                if body_base > 0 {
                    encode_fold(
                        encoder,
                        residual_h,
                        body_base,
                        &work.fold,
                        rigid.group_norm.span(0, 1),
                        true,
                    )?;
                }
                encoder.elements(
                    &PcgRigidGroupL1Args {
                        reduced: residual_h,
                        body_base: body_base as u32,
                        norm: rigid.group_norm.span(1, bodies),
                        count: bodies as u32,
                        seam_arena_count: 0,
                    },
                    bodies as u32,
                )?;
                if let Some(ref body)=rigid.body_factor {
                    reduction.encode_precond_cloth_fp32(encoder,inverse_diagonal,residual_h,preconditioned)?;
                    body.encode_apply_with_fault_target(encoder,residual_h,preconditioned,body_base as u32,Some(status),Some(factor_fault))?;
                } else {
                    reduction.encode_precond(encoder,factor,inverse_diagonal,residual_h,preconditioned)?;
                }
                if let Some(root) = roots { root.encode_project(encoder, preconditioned)?; }
                terms.encode_fill(encoder, residual_h, preconditioned, dim)?;
                encode_fold(encoder, products, terms.rows(), &work.fold, rz_next, false)?;
                encoder.elements(
                    &super::kernels::JointPcgGroupStopBetaArgs {
                        norm: rigid.group_norm.span(0, 1 + bodies),
                        initial: rigid.group_initial_device.handle(),
                        initial_all: err0,
                        groups: (1 + bodies) as u32,
                        cloth_present: (body_base > 0) as u32,
                        rz_next,
                        rz_previous,
                        beta,
                        relative,
                        status,
                        tolerance,
                        iteration,
                        max_iterations,
                        extent: 1,
                        count: 1,
                        seam_arena_count: 0,
                    },
                    1,
                )?;
                encoder.elements(
                    &super::kernels::FemPcgReducedDirectionArgs {
                        z: preconditioned,
                        p: direction,
                        beta,
                        status,
                        extent: dim as u32,
                        count: dim as u32,
                        seam_arena_count: 0,
                    },
                    dim as u32,
                )
            };
            // Only record warm interior iterations. The captured iteration
            // index only serves the cap comparison. The true cap is handled
            // above by the original uncaptured path with its real index.
            if stride_graph_enabled && iteration == 5 && stride_graph_capture_candidate(
                iteration, max_iterations, rigid.active_factor_borrowed.then(|| (
                    factor_iteration_started.elapsed().as_secs_f64() * 1000.0,
                    rigid.active_factor_prepare_ms,
                )),
            )
            {
                let before = device.counters().regions_deferred;
                let graph = device.record("pcg.joint.resident_stride_graph", &mut encode_stride)?;
                if device.counters().regions_deferred > before {
                    stride_graph = Some(graph);
                } else {
                    device.release(graph);
                    if stride_graph_receipt {
                        eprintln!("PPF_JOINT_STRIDE_GRAPH fallback=no_deferred_capture");
                    }
                }
            }
            let submitted = if let Some(graph) = stride_graph.as_ref() {
                match device.replay(graph, 1) {
                    Ok(diag) => {
                        stride_graph_replays += 1;
                        Ok(diag)
                    }
                    // CUDA rejects an allocator-generation mismatch before
                    // any graph launch. Other errors may have modified the
                    // recurrence and must not rerun the iteration.
                    Err(Fault::StaleRegion { .. }) => {
                        release_stride_graph!();
                        device.run("pcg.joint.resident_stride1", &mut encode_stride)
                    }
                    Err(error) => Err(error),
                }
            } else {
                device.run("pcg.joint.resident_stride1", &mut encode_stride)
            };
            if let Err(error) = submitted {
                release_stride_graph!();
                return Err(error.into());
            }
            if let Err(error) = work.scalar.download(device) {
                release_stride_graph!();
                return Err(error.into());
            }
            if rigid.body_factor.is_some() {
                if work.scalar.host()[slot::FACTOR_FAULT].to_bits()!=0 {
                        release_stride_graph!();
                        let mut owner=rigid.body_factor.take().expect("factor owner");
                        owner.free(device)?;
                        device.copy(rigid.xr.handle(),0,rigid.body_seed_x.handle(),0,dim*4)?;
                        device.copy(rigid.r.handle(),0,rigid.body_seed_r.handle(),0,dim*4)?;
                        rz=seed_direction!();
                        completed_iterations=iteration;
                        prefix_continues=false;
                        eprintln!("PPF_BODY_TREE_PCG fallback=cold_bj_restart reason=factor_fault");
                        break;
                }
            }
            if tet_receipt_pending { eprintln!("TET_FORCE_ONLY_RECEIPT active=true"); tet_receipt_pending=false; }
            if gather_receipt_pending { eprintln!("RESTRICT_GATHER_APPLY_RECEIPT active=true"); gather_receipt_pending=false; }
            if row_receipt_pending { eprintln!("JOINT_ROW_PARALLEL_RECEIPT active=true"); row_receipt_pending=false; }
            let residual = work.scalar.host()[slot::ALPHA_NOISE];
            let status = work.scalar.host()[slot::BETA_NOISE];
            if status == 0.0 {
                if let Some(trace)=lanczos.as_mut() {
                    trace.record(iteration,work.scalar.host()[slot::ALPHA],
                        work.scalar.host()[slot::BETA],work.scalar.host()[slot::P_AP]);
                }
            }
            if status == 1.0 {
                release_stride_graph!();
                let verified = verify_true_residual!(iteration, residual);
                if let Some(trace)=lanczos.as_mut() {
                    trace.finish(if verified < tolerance {"converged"}
                        else if iteration >= max_iterations || !verified.is_finite() {"verification_failed_exit"}
                        else {"true_residual_restart"});
                }
                if verified < tolerance {
                    finish!(Outcome::Converged, iteration, verified);
                }
                if rigid.body_factor.is_some() && animation_decision!() {
                    finish!(Outcome::AcceptedAnimation, iteration, verified);
                }
                if iteration >= max_iterations || !verified.is_finite() {
                    finish!(Outcome::MaxIterations, iteration, verified);
                }
                restart_from_true_residual!();
                completed_iterations = iteration;
                prefix_continues = false;
                break;
            }
            if status == 2.0 {
                release_stride_graph!();
                if let Some(trace)=lanczos.as_mut() {trace.finish("cap_or_nonfinite");}
                finish!(Outcome::MaxIterations, iteration, residual);
            }
            if status != 0.0 {
                release_stride_graph!();
                if let Some(trace)=lanczos.as_mut() {trace.finish("invalid_status");}
                return Err(Fault::Platform {
                    call: "pcg.joint.resident_stride1",
                    detail: "invalid status".to_owned(),
                }
                .into());
            }
            if iteration == max_iterations {
                finish!(Outcome::MaxIterations, iteration, residual);
            }
            let refresh_due = rigid.active_factor_borrowed && rigid.body_factor.is_some()
                && factor_iteration_started.elapsed().as_secs_f64() * 1000.0 >= rigid.active_factor_prepare_ms;
            let keep_borrowed = factor_refresh_trial.as_mut().is_some_and(|trial| {
                trial.defer(iteration, residual, tolerance, max_iterations, refresh_due)
            });
            if refresh_due && !keep_borrowed {
                release_stride_graph!();
                if let Some(trace)=lanczos.as_mut(){trace.finish("factor_refresh");}
                factor_refresh_pending=Some((iteration,residual));
                completed_iterations=iteration;
                prefix_continues=false;
                grouped_soft=None;
                break;
            }
        }
        release_stride_graph!();
        if batch_admitted && prefix_continues {
            let batch_row_active = row_parallel && joints.soft.is_some_and(|soft| soft.row_parallel_available());
            // Preserve the freshly downloaded eighth iteration, including RZ0.
            let mut transition = work.scalar.host().to_vec();
            let prefix_residual=transition[slot::ALPHA_NOISE];
            transition[slot::BREAK_FIRED] = f32::from_bits(0);
            transition[slot::FOLD_SCRATCH] = f32::from_bits(prefix_end);
            work.scalar.seed(device, &transition)?;
            ::log::info!("pcg.joint.resident_batch8 subset active tet={} soft={} controls={} extensions={}",
                op.has_tet_factor(), joints.soft.is_some(), joints.control_count, joints.extension_count);
            let direction = rigid.p.handle(); let applied = rigid.rp.handle();
            let residual_h = rigid.r.handle(); let preconditioned = rigid.z.handle();
            let rz_previous = work.scalar.span(slot::RZ0, 1); let pap = work.scalar.span(slot::P_AP, 1);
            let alpha = work.scalar.span(slot::ALPHA, 1); let rz_next = work.scalar.span(slot::RZ1, 1);
            let beta = work.scalar.span(slot::BETA, 1); let relative = work.scalar.span(slot::ALPHA_NOISE, 1);
            let status = work.scalar.span(slot::BETA_NOISE, 1); let products = terms.product_handle();
            let first_iteration = work.scalar.span(slot::BREAK_FIRED, 1);
            let current_iteration = work.scalar.span(slot::FOLD_SCRATCH, 1);
            let inject_step = match std::env::var("PPF_FACTOR_BATCH8_FAULT_INJECT") {
                Ok(value) => { let step=value.parse::<u32>().map_err(|_|Fault::Platform{call:"factor batch8 injection",detail:"expected 1, 4 or 8".into()})?;
                    if ![1,4,8].contains(&step) {return Err(Fault::Platform{call:"factor batch8 injection",detail:"requires step 1, 4 or 8".into()}.into());} if rigid.body_factor.is_some(){Some(step)}else{None} },
                Err(std::env::VarError::NotPresent)=>None,
                Err(_)=>return Err(Fault::Platform{call:"factor batch8 injection",detail:"invalid environment".into()}.into()),
            };
            let mut inject_rhs=Buffer::<f32>::none();
            if inject_step.is_some() {
                if let Err(error)=inject_rhs.size(device,dim,AllocLabel("factor_batch8.inject_rhs")) {let _=inject_rhs.free(device);return Err(error.into());}
                if let Err(error)=inject_rhs.write(device,0,&vec![f32::NAN;dim]) {let _=inject_rhs.free(device);return Err(error.into());}
            }
            let region_result = device.record("pcg.joint.resident_batch8", |encoder| unsafe {
                for substep in 1..=8 {
                    reduction.encode_prolong(encoder, direction, rigid.xv.handle())?;
                    op.encode_apply_batch8(
                        encoder,
                        rigid.xv.handle(),
                        rigid.mxv.handle(),
                        work.row_absolute.handle(),
                        work.row_curvature.handle(),
                        status,
                    )?;
                    reduction.encode_restrict(encoder, rigid.mxv.handle(), applied, false)?;
                    joints.encode_krylov_batch8(encoder, direction, applied, status, batch_row_active)?;
                    terms.encode_fill(encoder, direction, applied, dim)?;
                    encode_fold(encoder, products, terms.rows(), &work.fold, pap, false)?;
                    encoder.elements(
                        &super::kernels::FemPcgReducedBatchAlphaArgs {
                            rz: rz_previous,
                            pap,
                            alpha,
                            status,
                            extent: 1,
                            count: 1,
                            seam_arena_count: 0,
                        },
                        1,
                    )?;
                    encoder.elements(
                        &super::kernels::FemPcgReducedBatchAddIndirectArgs {
                            source: direction,
                            destination: rigid.xr.handle(),
                            coefficient: alpha,
                            sign: 1.0,
                            status,
                            extent: dim as u32,
                            count: dim as u32,
                            seam_arena_count: 0,
                        },
                        dim as u32,
                    )?;
                    encoder.elements(
                        &super::kernels::FemPcgReducedBatchAddIndirectArgs {
                            source: applied,
                            destination: residual_h,
                            coefficient: alpha,
                            sign: -1.0,
                            status,
                            extent: dim as u32,
                            count: dim as u32,
                            seam_arena_count: 0,
                        },
                        dim as u32,
                    )?;
                    if body_base > 0 {
                        encode_fold(
                            encoder,
                            residual_h,
                            body_base,
                            &work.fold,
                            rigid.group_norm.span(0, 1),
                            true,
                        )?;
                    }
                    encoder.elements(
                        &PcgRigidGroupL1Args {
                            reduced: residual_h,
                            body_base: body_base as u32,
                            norm: rigid.group_norm.span(1, bodies),
                            count: bodies as u32,
                            seam_arena_count: 0,
                        },
                        bodies as u32,
                    )?;
                    if let Some(ref body)=rigid.body_factor {
                        reduction.encode_precond_cloth_fp32(encoder,inverse_diagonal,residual_h,preconditioned)?;
                        let factor_rhs=if inject_step==Some(substep) {inject_rhs.handle()} else {residual_h};
                        body.encode_apply_with_fault_target(encoder,factor_rhs,preconditioned,body_base as u32,Some(status),Some(work.scalar.span(slot::FACTOR_FAULT,1)))?;
                        encoder.elements(&super::kernels::JointPcgBatch8FactorFaultArgs {
                            fault:work.scalar.span(slot::FACTOR_FAULT,1), status, first_iteration, current_iteration,
                            extent:1, count:1, seam_arena_count:0,
                        },1)?;
                    } else {
                    reduction.encode_precond(
                        encoder,
                        factor,
                        inverse_diagonal,
                        residual_h,
                        preconditioned,
                    )?;
                    }
                    terms.encode_fill(encoder, residual_h, preconditioned, dim)?;
                    encode_fold(encoder, products, terms.rows(), &work.fold, rz_next, false)?;
                    encoder.elements(
                        &super::kernels::JointPcgBatch8StopArgs {
                            norm: rigid.group_norm.span(0, 1 + bodies),
                            initial: rigid.group_initial_device.handle(),
                            initial_all: err0,
                            groups: (1 + bodies) as u32,
                            cloth_present: (body_base > 0) as u32,
                            rz_next,
                            rz_previous,
                            beta,
                            relative,
                            status,
                            tolerance,
                            first_iteration,
                            current_iteration,
                            max_iterations,
                            extent: 1,
                            count: 1,
                            seam_arena_count: 0,
                        },
                        1,
                    )?;
                    encoder.elements(
                        &super::kernels::FemPcgReducedDirectionArgs {
                            z: preconditioned,
                            p: direction,
                            beta,
                            status,
                            extent: dim as u32,
                            count: dim as u32,
                            seam_arena_count: 0,
                        },
                        dim as u32,
                    )?;
                }
                Ok(())
            });
            let region=match region_result {Ok(region)=>region,Err(error)=>{let _=inject_rhs.free(device);return Err(error.into());}};
            let mut injection_block_start=None;
            let result: super::scene::FatalResult<(u32,u32,f32)> = (|| {
                let mut previous = prefix_end;
                let mut last_residual=prefix_residual;
                let mut receipt_pending = true;
                while previous < max_iterations {
                    if rigid.active_factor_borrowed && rigid.body_factor.is_some() && factor_iteration_started.elapsed().as_secs_f64()*1000.0 >= rigid.active_factor_prepare_ms {
                        return Ok((3,previous,last_residual));
                    }
                    injection_block_start=Some(previous);
                    device.replay(&region, 1)?; work.scalar.download(device)?;
                    let current=work.scalar.host()[slot::FOLD_SCRATCH].to_bits();
                    let first=work.scalar.host()[slot::BREAK_FIRED].to_bits();
                    let residual=work.scalar.host()[slot::ALPHA_NOISE];
                    let status=work.scalar.host()[slot::BETA_NOISE];
                    let end=previous.saturating_add(8).min(max_iterations);
                    if work.scalar.host()[slot::FACTOR_FAULT].to_bits()!=0 {
                        if status != 3.0 || first != current || current <= previous || current > end {
                            return Err(Fault::Platform {call:"pcg.joint.batch8 factor fault",detail:"invalid factor fault latch".to_owned()}.into());
                        }
                        return Ok((2,current,residual));
                    }
                    if status == 0.0 {
                        if current != end || first != 0 || current <= previous {
                            return Err(Fault::Platform {call:"pcg.joint.batch8 counter", detail:"invalid counter".to_owned()}.into());
                        }
                        if receipt_pending { eprintln!("JOINT_BATCH8_APPLY_RECEIPT active=true first_iteration={} completed={} operator={}",previous+1,current,if batch_row_active {"row_guarded"} else {"guarded_original"}); receipt_pending=false; }
                        last_residual=residual;
                        previous=current;
                        if rigid.active_factor_borrowed && rigid.body_factor.is_some() && factor_iteration_started.elapsed().as_secs_f64()*1000.0 >= rigid.active_factor_prepare_ms { return Ok((3,current,residual)); }
                        continue;
                    }
                    if (status != 1.0 && status != 2.0) || first != current || current <= previous || current > end {
                        return Err(Fault::Platform {call:"pcg.joint.batch8 latch", detail:"invalid first stop".to_owned()}.into());
                    }
                    if receipt_pending { eprintln!("JOINT_BATCH8_APPLY_RECEIPT active=true first_iteration={} completed={} operator={}",previous+1,current,if batch_row_active {"row_guarded"} else {"guarded_original"}); }
                    return Ok((if status==1.0 {0} else {1}, first, residual));
                }
                Ok((1,max_iterations,1.0))
            })();
            device.release(region);
            let free_result=inject_rhs.free(device);
            let (result_kind, iteration, residual) = result?;
            free_result?;

            if result_kind==2 {
                let mut owner=rigid.body_factor.take().expect("factor fault requires owner");
                owner.free(device)?;
                device.copy(rigid.xr.handle(),0,rigid.body_seed_x.handle(),0,dim*4)?;
                device.copy(rigid.r.handle(),0,rigid.body_seed_r.handle(),0,dim*4)?;
                if let Some(step)=inject_step.filter(|step| work.scalar.host()[slot::FACTOR_FAULT].to_bits()==3 && injection_block_start.is_some_and(|start| iteration==start+*step)) {
                    let x=rigid_failure_read_f32(device,rigid.xr.handle(),dim)?;
                    let x0=rigid_failure_read_f32(device,rigid.body_seed_x.handle(),dim)?;
                    let r=rigid_failure_read_f32(device,rigid.r.handle(),dim)?;
                    let r0=rigid_failure_read_f32(device,rigid.body_seed_r.handle(),dim)?;
                    if !x.iter().zip(&x0).all(|(a,b)|a.to_bits()==b.to_bits()) || !r.iter().zip(&r0).all(|(a,b)|a.to_bits()==b.to_bits()) {
                        return Err(Fault::Platform{call:"factor batch8 injection",detail:"cold seed restoration failed".into()}.into());
                    }
                    eprintln!("PPF_FACTOR_BATCH8_ROLLBACK_SELF_CHECK passed step={} iterations={} region_released=true",step,iteration);
                    return Err(super::scene::Fatal::invariant("factor batch8 rollback self check complete; no commit"));
                }
                rz=seed_direction!();
                completed_iterations=iteration;
                grouped_soft=None;
                eprintln!("PPF_BODY_TREE_PCG fallback=cold_bj_restart reason=factor_fault");
            } else if result_kind==3 {
                factor_refresh_pending=Some((iteration,residual));
                completed_iterations=iteration;
                grouped_soft=None;
            } else {
                if result_kind==1 {finish!(Outcome::MaxIterations,iteration,residual);}
                let verified = verify_true_residual!(iteration,residual);
                if verified < tolerance {finish!(Outcome::Converged,iteration,verified);}
                if iteration >= max_iterations || !verified.is_finite() {finish!(Outcome::MaxIterations,iteration,verified);}
                restart_from_true_residual!(); completed_iterations=iteration;
            }
        }
    }

    // A refreshed solve continues through the classical recurrence below.
    // Batch its uninterrupted device work while retaining every host check.
    let coalesce_refreshed = auto_resident_active && factor_refresh_pending.is_some()
        && joints.soft.map_or(true, |soft| soft.frozen_restoring.is_some());
    if let Some((iteration,recursive))=factor_refresh_pending {
        let verified=verify_true_residual!(iteration,recursive);
        if verified<tolerance {finish!(Outcome::Converged,iteration,verified);}
        if recursive<tolerance && animation_decision!() {finish!(Outcome::AcceptedAnimation,iteration,verified);}
        if !verified.is_finite() || iteration>=max_iterations {finish!(Outcome::MaxIterations,iteration,verified);}
        device.copy(rigid.body_seed_x.handle(),0,rigid.xr.handle(),0,dim*4)?;
        device.copy(rigid.body_seed_r.handle(),0,rigid.r.handle(),0,dim*4)?;
        if let Some(mut owner)=rigid.body_factor.take(){owner.free(device)?;}
        rigid.active_factor_borrowed=false;
        rigid.active_factor_iteration_debt = None;
        let started=std::time::Instant::now();
        if let Some((engine,debug,reference))=prepare_body_tree(device,op,reduction,joints,roots,false,true)? {
            match engine {
                PreparedBodyTree::Factor(mut owner)=>{
                    if let Err(error)=device.run("body_factor.clear_fault.refresh",|encoder|owner.encode_clear_fault(encoder)) {
                        let _=owner.free(device);
                        return Err(error.into());
                    }
                    rigid.body_factor=Some(owner);
                    rigid.active_factor_prepare_ms=started.elapsed().as_secs_f64()*1000.0;
                    if iteration_debt_policy && rigid.active_factor_key.is_some() {
                        rigid.active_factor_fresh_prefix = Some(iteration);
                    }
                },
                PreparedBodyTree::Device(mut owner)=>{owner.free(device)?;return Err(Fault::Platform{call:"factor cache refresh",detail:"unexpected device inverse".into()}.into());},
                PreparedBodyTree::Host(_)=>return Err(Fault::Platform{call:"factor cache refresh",detail:"unexpected host tree".into()}.into()),
            }
            rigid.body_inverse_debug=debug;rigid.body_group_reference=reference;
        } else {
            rigid.active_factor_key=None;
            rigid.active_factor_prepare_ms=0.0;
        }
        rz=seed_direction!();
        if coalesce_refreshed && timing_receipt.is_some() {
            eprintln!("PPF_PCG_CLASSIC_TAIL coalesced=true completed_iterations={iteration} factor={}", rigid.body_factor.is_some());
        }
    }

    // Stage9: record fixed batch8 once; replay only observes the device counter.
    // Original scalar math/first-stop frozen fields and cache admission unchanged.
    if let Some(soft)=grouped_soft.filter(|view|view.resident_cache_ready()) {
        let mut seed=[0.0;slot::COUNT];seed[slot::RZ0]=rz;
        work.scalar.seed(device,&seed)?;
        let direction=rigid.p.handle();let applied=rigid.rp.handle();
        let residual_h=rigid.r.handle();let preconditioned=rigid.z.handle();
        let rz_previous=work.scalar.span(slot::RZ0,1);
        let pap=work.scalar.span(slot::P_AP,1);
        let alpha=work.scalar.span(slot::ALPHA,1);
        let error=work.scalar.span(slot::ERR,1);
        let rz_next=work.scalar.span(slot::RZ1,1);
        let beta=work.scalar.span(slot::BETA,1);
        let relative=work.scalar.span(slot::ALPHA_NOISE,1);
        let status=work.scalar.span(slot::BETA_NOISE,1);
        let products=terms.product_handle();
        let first_iteration=work.scalar.span(slot::BREAK_FIRED,1);
        let latched_abs_error=work.scalar.span(slot::BREAK_VALUE,1);
        let current_iteration=work.scalar.span(slot::FOLD_SCRATCH,1);
        let counters_before=device.counters();
        let mut replay_count=0u32;
        let region=device.record("pcg.fem.stage9.graph_batch8",|encoder| {
                unsafe {
                    for _subiteration in 0..8 {
                    op.encode_apply(encoder,direction,applied,
                        work.row_absolute.handle(),work.row_curvature.handle())?;
                    soft.encode_krylov_guarded(encoder,direction,applied,status)?;
                    terms.encode_fill(encoder,direction,applied,dim)?;
                    encode_fold_alpha_stage6(encoder,products,terms.rows(),
                        &work.fold,pap,rz_previous,alpha,status)?;
                    // Stage6 pure-FEM row fusion: preserve the original
                    // vec_add_scaled arithmetic and body-row preconditioner.
                    // The row writes r before the unchanged residual L1 fold.
                    // z and product are auxiliary on a latched stop, as before.
                    encoder.elements(&super::kernels::FemPcgReducedRowUpdateArgs {
                        direction,applied,alpha,status,iterate:rigid.xr.handle(),
                        residual:residual_h,vertex_body:reduction.staged.vbody.handle(),
                        cloth_offset:reduction.staged.cloth_off.handle(),
                        inverse_diagonal,preconditioned,
                        product:products,magnitude:terms.magnitude_handle(),
                        count:reduction.map.nrow as u32,seam_arena_count:0},
                        reduction.map.nrow as u32)?;
                    if reduction.staged.reduced_schwarz.active(){
                        // Fused BJ z/product were auxiliary. Overwrite BOTH with
                        // the selected model before the original r.z/stop fold.
                        reduction.encode_precond(encoder,factor,inverse_diagonal,residual_h,preconditioned)?;
                        terms.encode_fill(encoder,residual_h,preconditioned,dim)?;
                    }
                    let (error_source,error_count)=encode_fold_abs_level(
                        encoder,residual_h,body_base,&work.fold,error)?;
                    encode_fold_dual_stop_stage6(encoder,error_source,error_count,
                        products,terms.rows(),&work.fold,error_count,
                        fold_scratch_len(dim),error,rz_next,err0,rz_previous,
                        beta,relative,status,first_iteration,latched_abs_error,
                        current_iteration,tolerance,max_iterations,error_count==1)?;
                    encoder.elements(&super::kernels::FemPcgReducedDirectionArgs {
                        z:preconditioned,p:direction,beta,status,
                        extent:dim as u32,count:dim as u32,seam_arena_count:0},dim as u32)?;
                    }
                    Ok(())
                }
            })?;
        // After successful recording every result/error path releases the region
        // before finish/prolong. Recording itself never executes the iteration.
        let result:super::scene::FatalResult<(Outcome,u32,f32)>=(|| {
            let mut previous=0u32;
            while previous<max_iterations {
                device.replay(&region,1)?;
                replay_count+=1;
                work.scalar.download(device)?;
                reduction.staged.reduced_schwarz.check(device)?;
                let current=work.scalar.host()[slot::FOLD_SCRATCH].to_bits();
                let first=work.scalar.host()[slot::BREAK_FIRED].to_bits();
                let residual=work.scalar.host()[slot::ALPHA_NOISE];
                let status=work.scalar.host()[slot::BETA_NOISE];
                let expected_end=previous.saturating_add(8).min(max_iterations);
                if status==0.0 {
                    if current!=expected_end || first!=0 || current<=previous {
                        return Err(Fault::Platform {call:"pcg.fem.stage9 counter",
                            detail:"invalid active graph counter".to_owned()}.into());
                    }
                    previous=current;
                    continue;
                }
                if (status!=1.0 && status!=2.0) || first!=current ||
                    current<=previous || current>expected_end {
                    return Err(Fault::Platform {call:"pcg.fem.stage9 first stop",
                        detail:"invalid latched graph stop".to_owned()}.into());
                }
                rigid.group_current[0]=work.scalar.host()[slot::BREAK_VALUE];
                return Ok((if status==1.0{Outcome::Converged}else{Outcome::MaxIterations},first,residual));
            }
            Ok((Outcome::MaxIterations,max_iterations,1.0))
        })();
        device.release(region);
        let counters_after=device.counters();
        ::log::info!("pcg.fem.stage9.graph_batch8 proof: deferred={} fallback={} replays={} result_ok={}",
            counters_after.regions_deferred.saturating_sub(counters_before.regions_deferred),
            counters_after.regions_fallback.saturating_sub(counters_before.regions_fallback),
            replay_count,result.is_ok());
        let (outcome,iteration,residual)=result?;
        if !matches!(outcome, Outcome::Converged) {
            finish!(outcome,iteration,residual);
        }
        let true_residual = verify_true_residual!(iteration,residual);
        if true_residual < tolerance {
            finish!(Outcome::Converged,iteration,true_residual);
        }
        if iteration >= max_iterations || !true_residual.is_finite() {
            finish!(Outcome::MaxIterations,iteration,true_residual);
        }
        ::log::info!("pcg true-residual restart after graph iteration {}: recursive={:.9} true={:.9}",
            iteration,residual,true_residual);
        restart_from_true_residual!();
        completed_iterations = iteration;
        // The graph's stop latch is immutable for this run. Continue scalar
        // from the consumed count after releasing it, with the SAME M and cap.
        grouped_soft = None;
    }

    // Stage2: stride1, one ordered region and one scalar probe. Do not issue
    // another iteration before the original residual/cap stop is observed.
    // Seed/force/action0, nodal diagonal/action5 and finish remain upstream.
    if let Some(soft)=grouped_soft {
        let mut seed=[0.0;slot::COUNT];seed[slot::RZ0]=rz;
        work.scalar.seed(device,&seed)?;
        for iteration in 1..=max_iterations {
            let direction=rigid.p.handle();let applied=rigid.rp.handle();
            let residual_h=rigid.r.handle();let preconditioned=rigid.z.handle();
            let rz_previous=work.scalar.span(slot::RZ0,1);
            let pap=work.scalar.span(slot::P_AP,1);
            let alpha=work.scalar.span(slot::ALPHA,1);
            let error=work.scalar.span(slot::ERR,1);
            let rz_next=work.scalar.span(slot::RZ1,1);
            let beta=work.scalar.span(slot::BETA,1);
            let relative=work.scalar.span(slot::ALPHA_NOISE,1);
            let status=work.scalar.span(slot::BETA_NOISE,1);
            let products=terms.product_handle();
            device.run("pcg.fem.stage2.resident_stride1",|encoder| {
                unsafe {
                    op.encode_apply(encoder,direction,applied,
                        work.row_absolute.handle(),work.row_curvature.handle())?;
                    soft.encode_krylov(encoder,direction,applied)?;
                    terms.encode_fill(encoder,direction,applied,dim)?;
                    encode_fold(encoder,products,terms.rows(),&work.fold,pap,false)?;
                    encoder.elements(&super::kernels::FemPcgReducedAlphaArgs {
                        rz:rz_previous,pap,alpha,extent:1,count:1,seam_arena_count:0},1)?;
                    encoder.elements(&super::kernels::FemPcgReducedAddIndirectArgs {
                        source:direction,destination:rigid.xr.handle(),coefficient:alpha,
                        sign:1.0,extent:dim as u32,count:dim as u32,seam_arena_count:0},dim as u32)?;
                    encoder.elements(&super::kernels::FemPcgReducedAddIndirectArgs {
                        source:applied,destination:residual_h,coefficient:alpha,
                        sign:-1.0,extent:dim as u32,count:dim as u32,seam_arena_count:0},dim as u32)?;
                    encode_fold(encoder,residual_h,body_base,&work.fold,error,true)?;
                    // Pure original arithmetic; on a stopping iteration z/r.z
                    // are auxiliary only. Stop-beta does not consume them and
                    // conditional direction never changes p after a stop.
                    reduction.encode_precond(encoder,factor,inverse_diagonal,residual_h,preconditioned)?;
                    terms.encode_fill(encoder,residual_h,preconditioned,dim)?;
                    encode_fold(encoder,products,terms.rows(),&work.fold,rz_next,false)?;
                    encoder.elements(&super::kernels::FemPcgReducedStopBetaArgs {
                        error,initial_error:err0,rz_next,rz_previous,beta,relative,status,
                        tolerance,iteration,max_iterations,extent:1,count:1,seam_arena_count:0},1)?;
                    encoder.elements(&super::kernels::FemPcgReducedDirectionArgs {
                        z:preconditioned,p:direction,beta,status,
                        extent:dim as u32,count:dim as u32,seam_arena_count:0},dim as u32)
                }
            })?;
            work.scalar.download(device)?;
            reduction.staged.reduced_schwarz.check(device)?;
            rigid.group_current[0]=work.scalar.host()[slot::ERR];
            let residual=work.scalar.host()[slot::ALPHA_NOISE];
            let status=work.scalar.host()[slot::BETA_NOISE];
            if status==1.0 {
                let true_residual = verify_true_residual!(iteration,residual);
                if true_residual < tolerance {
                    finish!(Outcome::Converged,iteration,true_residual);
                }
                if iteration >= max_iterations || !true_residual.is_finite() {
                    finish!(Outcome::MaxIterations,iteration,true_residual);
                }
                ::log::info!("pcg true-residual restart after resident iteration {}: recursive={:.9} true={:.9}",
                    iteration,residual,true_residual);
                restart_from_true_residual!();
                completed_iterations = iteration;
                grouped_soft = None;
                break;
            }
            if status==2.0 {finish!(Outcome::MaxIterations,iteration,residual);}
            if status!=0.0 {return Err(Fault::Platform {
                call:"pcg.fem.stage2 probe",detail:"invalid reduced device status".to_owned()}.into());}
        }
        if grouped_soft.is_some() {
            finish!(Outcome::MaxIterations,max_iterations,1.0);
        }
    }

    let lazy_budget = max_iterations.saturating_sub(1).min(4);
    let mut lazy_attempted = false;
    let mut fused_factor_receipt_pending = timing_receipt.is_some();
    let graph_enabled = coalesce_refreshed && device.info().supports_deferred_regions
        && std::env::var_os("PPF_JOINT_CLASSIC_PRODUCT_GRAPH").is_some_and(|value| value == "1");
    let graph_receipt = graph_enabled
        && std::env::var_os("PPF_JOINT_CLASSIC_PRODUCT_GRAPH_RECEIPT").is_some_and(|value| value == "1");
    let graph_start = completed_iterations.saturating_add(5);
    let ordered_tail = cfg!(cuda_backend) && safe_krylov && coalesce_refreshed
        && roots.is_some() && grouped_soft.is_none() && !coalesce_mixed && !arithmetic_active
        && !reduction.staged.reduced_schwarz.active() && scalar_debug.is_none()
        && std::env::var_os("PPF_CUDA_CLASSIC_RZ_PACKET").is_some_and(|value| value == "1");
    let ordered_tail_trace = ordered_tail
        && std::env::var_os("PPF_CUDA_CLASSIC_RZ_PACKET_TRACE").is_some_and(|value| value == "1");
    let mut ordered_tail_packets = 0u32;
    let mut ordered_tail_skipped = 0u32;
    let mut ordered_tail_factor_restarts = 0u32;
    let mut product_graph = None::<D::Region>;
    let mut graph_replays = 0u32;
    // The packet names A and the product workspaces only. It must not be
    // applied to the cold BJ continuation after the fresh factor falls back.
    macro_rules! release_product_graph {
        () => {
            if let Some(region) = product_graph.take() {
                device.release(region);
                if graph_receipt {
                    eprintln!("PPF_JOINT_CLASSIC_PRODUCT_GRAPH released=true replays={graph_replays} roots={}", roots.is_some());
                }
            }
        };
    }
    let classic_result = (|| -> super::scene::FatalResult<Report> {
        for iteration in completed_iterations.saturating_add(1)..=max_iterations {
            let no_progress_eligible = input == RigidSolveInput::NewtonForce && reduction.staged.pcg_precision == crate::args::PcgPrecision::Fp32 && lock.is_none() && roots.is_none() && !reduction.map.needs_projection() && !reduction.staged.reduced_schwarz.active() && grouped_soft.is_none() && !coalesce_mixed && lazy_tree_mode.is_none() && !warm_eligible && !arithmetic_active && rigid.body_host_tree.is_none() && rigid.body_inverse.is_none() && rigid.body_factor.is_none();
            if !no_progress_eligible { no_progress.clear(); }
            if let Some(host_mode) = lazy_tree_mode {
                if !lazy_attempted && lazy_budget > 0 && iteration == lazy_budget + 1 {
                    no_progress.clear();
                    lazy_attempted = true;
                    let recursive = super::pdrd::worst_relative_residual(&rigid.group_current, &rigid.group_initial, err0);
                    let verified = verify_true_residual!(lazy_budget, recursive);
                    if verified < tolerance { finish!(Outcome::Converged, lazy_budget, verified); }
                    if recursive < tolerance && animation_decision!() { finish!(Outcome::AcceptedAnimation, lazy_budget, verified); }
                    if !verified.is_finite() { finish!(Outcome::MaxIterations, lazy_budget, verified); }
                    if let Some((engine,debug,reference)) = prepare_body_tree(device,op,reduction,joints,roots,host_mode,false)? {
                        match engine { PreparedBodyTree::Factor(owner)=>rigid.body_factor=Some(owner), PreparedBodyTree::Device(inv)=>rigid.body_inverse=Some(inv), PreparedBodyTree::Host(host)=>rigid.body_host_tree=Some(host) }
                        rigid.body_seed_x.size(device,dim,ppf_cts_compute::AllocLabel("body_tree.seed_x"))?;
                        rigid.body_seed_r.size(device,dim,ppf_cts_compute::AllocLabel("body_tree.seed_r"))?;
                        device.copy(rigid.body_seed_x.handle(),0,rigid.xr.handle(),0,dim*4)?;
                        device.copy(rigid.body_seed_r.handle(),0,rigid.r.handle(),0,dim*4)?;
                        rigid.body_inverse_debug=debug;
                        rigid.body_group_reference=reference;
                        eprintln!("PPF_BODY_TREE_PCG active=true apply={} schedule=lazy", if host_mode { "host" } else { "device" });
                    }
                    // Verification replaces r and canonicalizes x, even on refusal.
                    rz = seed_direction!();
                }
            }
            let prp = if let Some(soft)=grouped_soft {
                let direction=rigid.p.handle();
                let applied=rigid.rp.handle();
                let pap=work.scalar.span(slot::FOLD_SCRATCH,1);
                let products=terms.product_handle();
                device.run("pcg.fem.stage1.apply_dot",|encoder| {
                    // Ordered barriers separate all entries. Fold follows COMPLETE
                    // A+FEM Ap, including extension/restoring terms and native faults.
                    unsafe {
                        op.encode_apply(encoder,direction,applied,
                            work.row_absolute.handle(),work.row_curvature.handle())?;
                        soft.encode_krylov(encoder,direction,applied)?;
                        terms.encode_fill(encoder,direction,applied,dim)?;
                        encode_fold(encoder,products,terms.rows(),&work.fold,pap,false)
                    }
                })?;
                work.scalar.download(device)?;
                work.scalar.host()[slot::FOLD_SCRATCH]
            } else if fused_joint_product || coalesce_refreshed {
                let direction = rigid.p.handle();
                let applied = rigid.rp.handle();
                let pap = work.scalar.span(slot::FOLD_SCRATCH, 1);
                let mut encode_product = |encoder: &mut dyn Encoder| {
                    if let Some(root) = roots { root.encode_project(encoder, direction)?; }
                    reduction.encode_prolong(encoder, direction, rigid.xv.handle())?;
                    if tet_force_only {
                        op.encode_apply_force_only(encoder, rigid.xv.handle(), rigid.mxv.handle(),
                            work.row_absolute.handle(), work.row_curvature.handle())?;
                    } else {
                        op.encode_apply(encoder, rigid.xv.handle(), rigid.mxv.handle(),
                            work.row_absolute.handle(), work.row_curvature.handle())?;
                    }
                    reduction.encode_restrict(encoder, rigid.mxv.handle(), applied, allow_gather || coalesce_refreshed)?;
                    joints.encode_krylov(encoder, direction, applied, row_parallel)?;
                    if let Some(root) = roots { root.encode_project(encoder, applied)?; }
                    terms.encode_fill(encoder, direction, applied, dim)?;
                    encode_fold(encoder, terms.product_handle(), terms.rows(), &work.fold, pap, false)
                };
                // Only the fixed A -> pAp packet is captured. Alpha and all
                // convergence, preconditioner, and retry decisions stay on the host.
                if graph_enabled && rigid.body_factor.is_some() && iteration == graph_start
                    && max_iterations.saturating_sub(iteration) >= 4
                {
                    let before = device.counters().regions_deferred;
                    let region = device.record("pcg.joint.classic_product_graph", &mut encode_product)?;
                    if device.counters().regions_deferred > before {
                        product_graph = Some(region);
                    } else {
                        device.release(region);
                        if graph_receipt {
                            eprintln!("PPF_JOINT_CLASSIC_PRODUCT_GRAPH fallback=no_deferred_capture");
                        }
                    }
                }
                if let Some(region) = product_graph.as_ref() {
                    match device.replay(region, 1) {
                        Ok(_) => graph_replays += 1,
                        // The stale-generation check precedes all dispatches.
                        Err(Fault::StaleRegion { .. }) => {
                            release_product_graph!();
                            device.run("pcg.joint.apply_dot", &mut encode_product)?;
                        }
                        Err(error) => return Err(error.into()),
                    }
                } else {
                    device.run("pcg.joint.apply_dot", &mut encode_product)?;
                }
                if fused_factor_receipt_pending { eprintln!("PPF_FUSED_PRODUCT_EXECUTED factor_owner={}", rigid.body_factor.is_some()); fused_factor_receipt_pending=false; }
                if tet_receipt_pending { eprintln!("TET_FORCE_ONLY_RECEIPT active=true"); tet_receipt_pending=false; }
                if gather_receipt_pending { eprintln!("RESTRICT_GATHER_APPLY_RECEIPT active=true"); gather_receipt_pending=false; }
                if row_receipt_pending { eprintln!("JOINT_ROW_PARALLEL_RECEIPT active=true"); row_receipt_pending=false; }
                work.scalar.download(device)?;
                work.scalar.host()[slot::FOLD_SCRATCH]
            } else {
                apply_reduced_tangent!(rigid.p.handle(), rigid.rp.handle());
                terms.dot(device, rigid.p.handle(), rigid.rp.handle(), dim, &work.fold, &mut work.scalar)?
            };
            // A ZERO CURVATURE DIVIDES BY 1 RATHER THAN BY ZERO, which keeps the
            // step finite and lets the residual test below decide. It is NOT treated
            // as a breakdown here: the reduced operator is projected on both sides,
            // so an exactly zero `p^T R p` is a direction the constraints have
            // removed rather than an indefinite matrix.
            let alpha = if let Some(debug)=scalar_debug.as_mut(){let p=debug.dot(device,rigid.p.handle(),rigid.rp.handle(),dim)?;(debug.rz / if p!=0. {p}else{1.}) as f32}else{rz / if prp != 0.0 { prp } else { 1.0 }};
            if arithmetic_active && (!prp.is_finite() || prp <= 0.0 || !alpha.is_finite() || alpha <= 0.0) {
                downgrade_and_restart!("search_curvature", iteration, f32::INFINITY);
                continue;
            }
            // Read-only samples are kept in memory; no file is written on success.
            if failure_capture_dir.is_some() && (iteration == 1 || iteration == max_iterations) {
                let sample = match rigid_failure_vectors(device, failure_handles, dim, iteration, rz, prp, alpha) {
                    Ok(value) => value,
                    Err(error) => serde_json::json!({"capture_error":error.to_string()}),
                };
                if iteration == 1 { failure_first = sample.clone(); }
                if iteration == max_iterations { failure_cap_before = sample; }
            }
            if coalesce_refreshed {
                let direction = rigid.p.handle();
                let iterate = rigid.xr.handle();
                let applied = rigid.rp.handle();
                let residual = rigid.r.handle();
                group_l1_after(device, residual, &mut rigid.group_norm, &work.fold,
                    body_base, bodies, &mut rigid.group_current, |encoder| {
                        encode_add_scaled(encoder, direction, iterate, alpha, dim)?;
                        encode_add_scaled(encoder, applied, residual, -alpha, dim)?;
                        if let Some(root) = roots { root.encode_project(encoder, residual)?; }
                        Ok(())
                    })?;
            } else {
                device.run("pdrd.cg.update", |encoder| {
                    // Safety: the four buffers are borrowed for the whole call and the
                    // two updates touch disjoint destinations.
                    unsafe {
                        encode_add_scaled(encoder, rigid.p.handle(), rigid.xr.handle(), alpha, dim)?;
                        encode_add_scaled(encoder, rigid.rp.handle(), rigid.r.handle(), -alpha, dim)?;
                        if grouped_soft.is_some() {
                            encode_fold(encoder,rigid.r.handle(),body_base,&work.fold,
                                rigid.group_norm.span(0,1),true)?;
                        }
                        Ok(())
                    }
                })?;

                // Before the residual is measured: the norm has to be taken over the
                // PROJECTED residual, or a component the constraint removes is counted
                // against the tolerance and the solve never reaches it.
                project_tangent!(rigid.r.handle());
                // Safety: every handle names a live allocation for the whole call.
                if grouped_soft.is_some() {
                    rigid.group_norm.download(device)?;
                    rigid.group_current[0]=rigid.group_norm.host()[0];
                } else if coalesce_mixed {
                    if timing_receipt.is_some() { coalesced_iterations += 1; }
                    let residual_h = rigid.r.handle();
                    let preconditioned = rigid.z.handle();
                    let norm_cloth = rigid.group_norm.span(0, 1);
                    let norm_bodies = rigid.group_norm.span(1, bodies);
                    let dot_slot = rigid.group_norm.span(1 + bodies, 1);
                    let cross_slot = rigid.group_norm.span(2 + bodies, 1);
                    let products = terms.product_handle();
                    let arithmetic = arithmetic_active.then_some(record_arithmetic);
                    if arithmetic_active {
                        arithmetic_applies += 1;
                    }
                    device.run("pcg.rigid.group_l1_precond_dot", |encoder| {
                        unsafe {
                            if body_base > 0 {
                                encode_fold(encoder, residual_h, body_base, &work.fold, norm_cloth, true)?;
                            }
                            encoder.elements(&PcgRigidGroupL1Args {
                                reduced: residual_h, body_base: body_base as u32,
                                norm: norm_bodies, count: bodies as u32, seam_arena_count: 0,
                            }, bodies as u32)?;
                            reduction.encode_precond_selected(encoder, factor, inverse_diagonal,
                                residual_h, preconditioned, arithmetic)?;
                            terms.encode_fill(encoder, residual_h, preconditioned, dim)?;
                            encode_fold(encoder, products, terms.rows(), &work.fold, dot_slot, false)?;
                            if arithmetic_active {
                                terms.encode_fill(encoder, preconditioned, rigid.rp.handle(), dim)?;
                                encode_fold(encoder, products, terms.rows(), &work.fold, cross_slot, false)?;
                            }
                            Ok(())
                        }
                    })?;
                    rigid.group_norm.download(device)?;
                    let host = rigid.group_norm.host();
                    rigid.group_current[0] = if body_base > 0 { host[0] } else { 0.0 };
                    rigid.group_current[1..].copy_from_slice(&host[1..1 + bodies]);
                } else {
                    unsafe {
                        group_l1_device(device,rigid.r.handle(),&mut rigid.group_norm,
                            &work.fold,body_base,bodies,&mut rigid.group_current)
                    }?;
                }
            }
            let mut residual = super::pdrd::worst_relative_residual(
                &rigid.group_current,
                &rigid.group_initial,
                err0,
            );
            if residual < tolerance {
                release_product_graph!();
                let recursive_residual = residual;
                residual = verify_true_residual!(iteration,recursive_residual);
                if residual < tolerance {
                    finish!(Outcome::Converged, iteration, residual);
                }
                if iteration >= 1 && animation_decision!() {finish!(Outcome::AcceptedAnimation,iteration,residual);}
                if iteration < max_iterations && residual.is_finite() {
                    ::log::info!("pcg true-residual restart after scalar iteration {}: recursive={:.9} true={:.9}",
                        iteration,recursive_residual,residual);
                    if arithmetic_active && (true_residual_checks >= 2 || iteration >= arithmetic_limit) {
                        arithmetic_active = false;
                        arithmetic_downgrade = Some(if iteration >= arithmetic_limit {
                            "reserved_fp32_budget"
                        } else {
                            "repeated_false_convergence"
                        });
                    }
                    restart_from_true_residual!();
                    if no_progress_eligible && !arithmetic_active && rigid.body_host_tree.is_none() && rigid.body_inverse.is_none() && rigid.body_factor.is_none() {
                        if let Some(best) = no_progress.observe(residual) {
                            ::log::info!("pcg NoProgress reason=failed_true_restart_budget patience=16 best_verified={:.9e} last_verified={:.9e}", best, residual);
                            finish!(Outcome::NoProgress, iteration, residual);
                        }
                    }
                    continue;
                }
            } else if arithmetic_active && !residual.is_finite() {
                downgrade_and_restart!("recursive_residual", iteration, residual);
                continue;
            } else if arithmetic_active && iteration >= arithmetic_limit && iteration < max_iterations {
                downgrade_and_restart!("reserved_fp32_budget", iteration, residual);
                continue;
            }
            if iteration >= max_iterations || !residual.is_finite() {
                if group_debug {eprintln!("PPF_BODY_GROUP_CAP solve={} iteration={} residual={:.9e}",scalar_debug_id.unwrap(),iteration,residual);return Err(Fault::Platform{call:"body group diagnostic",detail:"cap; intentional no-commit".into()}.into());}
                if let Some(debug)=scalar_debug.as_ref(){debug.receipt(iteration,residual,f32::NAN,"cap");return Err(Fault::Platform{call:"scalar64 diagnostic",detail:"cap; intentional no-commit".into()}.into());}
                if let Some(dir) = failure_capture_dir.as_deref() {
                    let frozen = match rigid_failure_frozen_system(device, op, reduction, factor,
                        inverse_diagonal, failure_handles, &rigid.zero, b, joints, roots) {
                        Ok(value) => value,
                        Err(error) => serde_json::json!({"capture_error":error.to_string()}),
                    };
                    let capture = serde_json::json!({
                        "schema":"ppf.rigid-pcg-failure.v1",
                        "scope":"failed uncommitted Newton solve; original MaxIterations refusal retained",
                        "outcome":"MaxIterations","iteration":iteration,"max_iterations":max_iterations,
                        "relative_residual_f32":residual,"relative_residual_bits":residual.to_bits(),
                        "tolerance_f32":tolerance,"err0_f32":err0,"err0_bits":err0.to_bits(),
                        "group_initial_f32":rigid_failure_values(&rigid.group_initial),
                        "group_current_f32":rigid_failure_values(&rigid.group_current),
                        "previous_beta_f32":failure_previous_beta,
                        "previous_beta_bits":failure_previous_beta.map(f32::to_bits),
                        "physical_start_s":joints.start_time_s,"sample_time_s":joints.sample_time_s,"dt_s":joints.dt,
                        "dim":dim,"bodies":bodies,"body_base":body_base,"rows":op.rows(),
                        "first_iteration_before_update":failure_first,
                        "cap_iteration_before_update":failure_cap_before,
                        "frozen_after_failed_update":frozen,
                    });
                    if let Err(error) = rigid_failure_write(dir, &capture) {
                        eprintln!("RIGID_PCG_FAILURE_CAPTURE_ERROR {error}");
                    }
                }
                finish!(Outcome::MaxIterations, iteration, residual);
            }

            let (rz_next, cross) = if coalesce_mixed && grouped_soft.is_none() {
                let host = rigid.group_norm.host();
                (host[1 + bodies], arithmetic_active.then(|| host[2 + bodies]))
            } else {
                let packet_root = roots.filter(|root| root.count != 0 && ordered_tail && terms.rows() != 0
                    && rigid.body_factor.is_some() && rigid.body_inverse.is_none()
                    && rigid.body_host_tree.is_none());
                // The original factor fault check runs before any speculative
                // packet work. Failed factors go to the same cold Jacobi seed.
                precondition_current!(packet_root.is_none());
                if factor_restarted {
                    if ordered_tail_trace { ordered_tail_factor_restarts += 1; }
                    release_product_graph!();
                    no_progress.clear();
                    factor_restarted=false;
                    rz=seed_direction!();
                    continue;
                }
                let rz_next = if let Some(root) = packet_root {
                    let value = terms.ordered_root_dot(device, root, rigid.r.handle(),
                        rigid.z.handle(), dim, &work.fold, &mut work.scalar)?;
                    if ordered_tail_trace { ordered_tail_packets += 1; }
                    value
                } else {
                    if ordered_tail_trace { ordered_tail_skipped += 1; }
                    terms.dot(device, rigid.r.handle(), rigid.z.handle(), dim, &work.fold, &mut work.scalar)?
                };
                let cross = if arithmetic_active {
                    Some(terms.dot(device, rigid.z.handle(), rigid.rp.handle(), dim, &work.fold, &mut work.scalar)?)
                } else {
                    None
                };
                (rz_next, cross)
            };
            // One flexible-CG orthogonalization against the previous direction:
            // choose beta so (z_new + beta*p_old) is A-conjugate to p_old.
            let beta = if let Some(debug)=scalar_debug.as_mut(){let next=debug.dot(device,rigid.r.handle(),rigid.z.handle(),dim)?;let beta=(next/debug.rz) as f32;debug.rz=next;beta}else if let Some(cross) = cross { -cross / prp } else { rz_next / rz };
            if arithmetic_active && (!rz_next.is_finite() || rz_next <= 0.0 || !beta.is_finite()) {
                downgrade_and_restart!("flexible_direction", iteration, residual);
                continue;
            }
            if failure_capture_dir.is_some() { failure_previous_beta = Some(beta); }
            if let Some(root) = roots.filter(|_| coalesce_refreshed) {
                let z = rigid.z.handle();
                let p = rigid.p.handle();
                device.run("pdrd.cg.direction", |encoder| {
                    encode_combine_raw(encoder, z, p, p, 1.0, beta, dim)?;
                    root.encode_project(encoder, p)
                })?;
            } else {
                combine(
                    device,
                    rigid.z.handle(),
                    rigid.p.handle(),
                    rigid.p.handle(),
                    1.0,
                    beta,
                    dim,
                )?;
                // The new search direction in its own right, as at the seed.
                project_tangent!(rigid.p.handle());
            }
            rz = rz_next;
        }

        let _ = work;
        finish!(Outcome::MaxIterations, max_iterations, 1.0);
    })();
    release_product_graph!();
    if ordered_tail_trace {
        eprintln!("PPF_CUDA_CLASSIC_RZ_PACKET packets={ordered_tail_packets} skipped={ordered_tail_skipped} factor_restarts={ordered_tail_factor_restarts} typed_fault={}", classic_result.is_err());
    }
    classic_result
}

// Optional failure-only diagnostics for the actual two-body fixture. No
// decisions, tolerances, force law, solve kernel or solution vector are changed.
#[derive(Clone,Copy)]
struct RigidFailureHandles {
    r:ppf_cts_compute::Handle,z:ppf_cts_compute::Handle,p:ppf_cts_compute::Handle,
    ap:ppf_cts_compute::Handle,rhs:ppf_cts_compute::Handle,x:ppf_cts_compute::Handle,
    blocks:Option<ppf_cts_compute::Handle>,
}
fn rigid_failure_read_f32(device:&mut impl Device,handle:ppf_cts_compute::Handle,n:usize)->Result<Vec<f32>,Fault> {
    let mut values=vec![0.0f32;n];
    if n==0{return Ok(values);}
    let bytes=unsafe {std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(),n*4)};
    device.read(handle,0,bytes)?;Ok(values)
}
fn rigid_failure_read_u32(device:&mut impl Device,handle:ppf_cts_compute::Handle,n:usize)->Result<Vec<u32>,Fault> {
    let mut values=vec![0u32;n];
    if n==0{return Ok(values);}
    let bytes=unsafe {std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(),n*4)};
    device.read(handle,0,bytes)?;Ok(values)
}
fn rigid_failure_values(values:&[f32])->serde_json::Value {
    serde_json::json!({"f32":values,"u32_bits":values.iter().copied().map(f32::to_bits).collect::<Vec<_>>()})
}
fn rigid_failure_dot64(a:&[f32],b:&[f32])->f64 {
    a.iter().zip(b).map(|(x,y)|f64::from(*x)*f64::from(*y)).sum()
}
fn rigid_failure_vectors(device:&mut impl Device,handles:RigidFailureHandles,dim:usize,iteration:u32,
                         rz:f32,pap:f32,alpha:f32)->Result<serde_json::Value,Fault> {
    let r=rigid_failure_read_f32(device,handles.r,dim)?;
    let z=rigid_failure_read_f32(device,handles.z,dim)?;
    let p=rigid_failure_read_f32(device,handles.p,dim)?;
    let ap=rigid_failure_read_f32(device,handles.ap,dim)?;
    Ok(serde_json::json!({"stage":"before_update","iteration":iteration,
        "r":rigid_failure_values(&r),"z":rigid_failure_values(&z),
        "p":rigid_failure_values(&p),"Ap":rigid_failure_values(&ap),
        "rhs_reduced_projected":rigid_failure_values(&rigid_failure_read_f32(device,handles.rhs,dim)?),
        "x_reduced_affine":rigid_failure_values(&rigid_failure_read_f32(device,handles.x,dim)?),
        "rz_f32":rz,"pap_f32":pap,"alpha_f32":alpha,
        "rz_f32_bits":rz.to_bits(),"pap_f32_bits":pap.to_bits(),"alpha_f32_bits":alpha.to_bits(),
        "rz_f64_same_vectors":rigid_failure_dot64(&r,&z),"pap_f64_same_vectors":rigid_failure_dot64(&p,&ap),
        "r_l1_f64":r.iter().map(|x|f64::from(x.abs())).sum::<f64>()}))
}
fn rigid_failure_csr(device:&mut impl Device,offset:ppf_cts_compute::Handle,index:ppf_cts_compute::Handle,
                     value:ppf_cts_compute::Handle,rows:usize)->Result<serde_json::Value,Fault> {
    let offsets=rigid_failure_read_u32(device,offset,rows+1)?;
    let count=*offsets.last().unwrap() as usize;
    Ok(serde_json::json!({"offset_u32":offsets,"index_u32":rigid_failure_read_u32(device,index,count)?,
        "block_3x3_column_major":rigid_failure_values(&rigid_failure_read_f32(device,value,9*count)?)}))
}
unsafe fn animation_motion_model<D:Device>(device:&mut D,reduction:&super::pdrd::Reduction<'_>,ctx:InertialGuessContext,mask:ppf_cts_compute::Handle,roots:Option<super::roots::View>,dt:f32)->Result<Option<super::pcg_motion_stop::MotionModel>,Fault>{
    let n=reduction.map.nrow;
    let mut props=vec![crate::data::VertexProp::default();n];
    if n>0 {let bytes=std::slice::from_raw_parts_mut(props.as_mut_ptr().cast::<u8>(),n*std::mem::size_of::<crate::data::VertexProp>());device.read(ctx.prop,0,bytes)?;}
    let mut pulls=vec![crate::data::PullPair::default();ctx.pull_count];
    if !pulls.is_empty(){let bytes=std::slice::from_raw_parts_mut(pulls.as_mut_ptr().cast::<u8>(),pulls.len()*std::mem::size_of::<crate::data::PullPair>());device.read(ctx.pull,0,bytes)?;}
    let mut active:Vec<bool>=props.iter().map(|p|p.fix_index==0).collect();
    for p in &pulls {let i=p.index as usize;if i<n {active[i]=false;}}
    let masses:Vec<f32>=props.iter().map(|p|p.mass).collect();
    let raw_mask=rigid_failure_read_u32(device,mask,n)?;
    let Some(rows)=warm_start_policy::reduced_free_rows(reduction.map,&raw_mask) else{return Ok(None)};
    let mut free:Vec<bool>=rows.iter().map(|&x|x!=0).collect();
    if let Some(root)=roots {let mut targets=vec![crate::joint_root_motion_gpu::RootTargetGpu::default();root.count as usize];
        if !targets.is_empty(){let bytes=std::slice::from_raw_parts_mut(targets.as_mut_ptr().cast::<u8>(),targets.len()*std::mem::size_of::<crate::joint_root_motion_gpu::RootTargetGpu>());device.read(root.targets,0,bytes)?;}
        for t in targets {if t.enabled!=0 {let o=reduction.map.body_base+6*(t.body as usize);if o+6>free.len(){return Ok(None)}free[o..o+6].fill(false);}}
    }
    let prot=rigid_failure_read_f32(device,reduction.staged.prot.handle(),3*n)?;
    Ok(super::pcg_motion_stop::MotionModel::build(reduction.map,&prot,&masses,&active,&free,dt))
}

unsafe fn inertial_probe_columns<D:Device>(device:&mut D,op:&Operator,reduction:&mut super::pdrd::Reduction<'_>,joints:super::joints::View,zero:&[f32])->Result<(Vec<f32>,Vec<f32>),Fault> {
    let dim=reduction.map.dim; let rows=op.rows() as usize;
    let mut probe=Buffer::<f32>::default(); let mut out=Buffer::<f32>::default();
    let mut full=Buffer::<f32>::default(); let mut applied=Buffer::<f32>::default();
    let mut absolute=Buffer::<f32>::default(); let mut curvature=Buffer::<f32>::default();
    let result=(||->Result<(Vec<f32>,Vec<f32>),Fault>{
        probe.size(device,dim,AllocLabel("inertial.probe.column"))?; out.size(device,dim,AllocLabel("inertial.probe.out"))?;
        full.size(device,3*rows,AllocLabel("inertial.probe.full"))?; applied.size(device,3*rows,AllocLabel("inertial.probe.applied"))?;
        absolute.size(device,rows,AllocLabel("inertial.probe.absolute"))?; curvature.size(device,rows,AllocLabel("inertial.probe.curvature"))?;
        let mut columns=Vec::with_capacity(dim*dim);
        for c in 0..dim { let mut unit=vec![0f32;dim]; unit[c]=1.; probe.write(device,0,&unit)?;
            reduction.prolong(device,probe.handle(),full.handle())?;
            op.apply(device,full.handle(),applied.handle(),absolute.handle(),curvature.handle())?;
            reduction.restrict(device,applied.handle(),out.handle(),zero)?;
            joints.add(device,1,probe.handle(),out.handle())?;
            reduction.project_bodies(device,out.handle())?;
            columns.extend(rigid_failure_read_f32(device,out.handle(),dim)?);
        }
        probe.write(device,0,&vec![0f32;dim])?;
        reduction.prolong(device,probe.handle(),full.handle())?;
        op.apply(device,full.handle(),applied.handle(),absolute.handle(),curvature.handle())?;
        reduction.restrict(device,applied.handle(),out.handle(),zero)?;
        joints.add(device,1,probe.handle(),out.handle())?;
        reduction.project_bodies(device,out.handle())?;
        let a0=rigid_failure_read_f32(device,out.handle(),dim)?;
        Ok((columns,a0))
    })();
    let mut cleanup_error=None;
    for b in [&mut probe,&mut out,&mut full,&mut applied,&mut absolute,&mut curvature] {if let Err(e)=b.free(device){if cleanup_error.is_none(){cleanup_error=Some(e);}}}
    match result {Err(e)=>Err(e),Ok(v)=>{if let Some(e)=cleanup_error {return Err(e)} Ok(v)}}
}

unsafe fn rigid_failure_frozen_system<D:Device>(device:&mut D,op:&Operator,
    reduction:&mut super::pdrd::Reduction<'_>,factor:ppf_cts_compute::Handle,
    inverse:ppf_cts_compute::Handle,handles:RigidFailureHandles,zero:&[f32],full_rhs:ppf_cts_compute::Handle,
    joints:super::joints::View,roots:Option<super::roots::View>)->Result<serde_json::Value,Fault> {
    let dim=reduction.map.dim;let rows=op.rows() as usize;
    // Separate scratch: no existing Krylov vector, force, fitted pose or PCG
    // reduction scratch is used as a diagnostic destination.
    let mut probe=Buffer::<f32>::default();let mut out=Buffer::<f32>::default();
    let mut full=Buffer::<f32>::default();let mut applied=Buffer::<f32>::default();
    let mut absolute=Buffer::<f32>::default();let mut curvature=Buffer::<f32>::default();
    let fault_count=joints.control_count.max(joints.extension_count) as usize;
    let fault_before=if fault_count>0 {Some(rigid_failure_read_u32(device,joints.fault,fault_count)?)}else{None};
    let result=(||->Result<serde_json::Value,Fault>{
        probe.size(device,dim,AllocLabel("pcg.failure.probe"))?;
        out.size(device,dim,AllocLabel("pcg.failure.out"))?;
        full.size(device,3*rows,AllocLabel("pcg.failure.full"))?;
        applied.size(device,3*rows,AllocLabel("pcg.failure.applied"))?;
        absolute.size(device,rows,AllocLabel("pcg.failure.absolute"))?;
        curvature.size(device,rows,AllocLabel("pcg.failure.curvature"))?;
        let mut a_columns=Vec::new();let mut m_columns=Vec::new();let mut q_columns=Vec::new();
        for c in 0..dim {
            let mut unit=vec![0.0f32;dim];unit[c]=1.0;probe.write(device,0,&unit)?;
            reduction.project_bodies(device,probe.handle())?;
            if let Some(root)=roots {root.project(device,probe.handle())?;}
            q_columns.push(rigid_failure_values(&rigid_failure_read_f32(device,probe.handle(),dim)?));
            reduction.prolong(device,probe.handle(),full.handle())?;
            op.apply(device,full.handle(),applied.handle(),absolute.handle(),curvature.handle())?;
            reduction.restrict(device,applied.handle(),out.handle(),zero)?;
            joints.add(device,1,probe.handle(),out.handle())?;
            reduction.project_bodies(device,out.handle())?;
            if let Some(root)=roots {root.project(device,out.handle())?;}
            a_columns.push(rigid_failure_values(&rigid_failure_read_f32(device,out.handle(),dim)?));
            reduction.apply_precond(device,factor,inverse,probe.handle(),out.handle())?;
            reduction.project_bodies(device,out.handle())?;
            if let Some(root)=roots {root.project(device,out.handle())?;}
            m_columns.push(rigid_failure_values(&rigid_failure_read_f32(device,out.handle(),dim)?));
        }
        // Affine apply mirrors the original seed/residual operator exactly:
        // input is NOT projected, because its root particular rows matter.
        let x=rigid_failure_read_f32(device,handles.x,dim)?;
        probe.write(device,0,&x)?;
        reduction.prolong(device,probe.handle(),full.handle())?;
        op.apply(device,full.handle(),applied.handle(),absolute.handle(),curvature.handle())?;
        reduction.restrict(device,applied.handle(),out.handle(),zero)?;
        joints.add(device,1,probe.handle(),out.handle())?;
        reduction.project_bodies(device,out.handle())?;
        if let Some(root)=roots {root.project(device,out.handle())?;}
        combine(device,handles.rhs,out.handle(),probe.handle(),1.0,-1.0,dim)?;
        reduction.project_bodies(device,probe.handle())?;
        if let Some(root)=roots {root.project(device,probe.handle())?;}
        let true_residual=rigid_failure_read_f32(device,probe.handle(),dim)?;
        let dynamic=if let Some(d)=op.dynamic {
            rigid_failure_csr(device,d.offset,d.index,d.value,rows)?
        }else{serde_json::Value::Null};
        let fixed=rigid_failure_csr(device,op.fixed.offset,op.fixed.index,op.fixed.value,rows)?;
        let root_records=if let Some(root)=roots {
            let mut bytes=vec![0u8;root.count as usize*std::mem::size_of::<crate::joint_root_motion_gpu::RootTargetGpu>()];
            device.read(root.targets,0,&mut bytes)?;
            serde_json::json!({"count":root.count,"stride":64,"bytes_u8":bytes})
        }else{serde_json::Value::Null};
        let mut state=vec![0u8;joints.bodies as usize*std::mem::size_of::<crate::data::PdrdRigidState>()];
        device.read(joints.state,0,&mut state)?;
        let old_block_jacobi_not_built=reduction.staged.reduced_schwarz.active();
        let blocks=if old_block_jacobi_not_built {serde_json::Value::Null}else if let Some(h)=handles.blocks {
            rigid_failure_values(&rigid_failure_read_f32(device,h,36*reduction.map.n_bodies)?)
        }else{serde_json::Value::Null};
        Ok(serde_json::json!({"matrix_layout":"array of exact device-applied columns",
            "A_Q_reduced_operator_Q_columns":a_columns,"M_Q_preconditioner_inverse_Q_columns":m_columns,
            "Q_columns":q_columns,"root_records":root_records,"rigid_state_bytes_u8":state,
            "assembled_precond_6x6_row_major":blocks,
            "factor_6x6_row_major":if old_block_jacobi_not_built {serde_json::Value::Null}else{rigid_failure_values(&rigid_failure_read_f32(device,factor,36*reduction.map.n_bodies)?)},
            "old_block_jacobi_not_built":old_block_jacobi_not_built,
            "inverse_vertex_diagonal":rigid_failure_values(&rigid_failure_read_f32(device,inverse,9*rows)?),
            "full_rhs":rigid_failure_values(&rigid_failure_read_f32(device,full_rhs,3*rows)?),
            "diagonal_3x3_column_major":rigid_failure_values(&rigid_failure_read_f32(device,op.diagonal,9*rows)?),
            "fixed_csr":fixed,"dynamic_csr":dynamic,
            "true_projected_affine_residual_f32":rigid_failure_values(&true_residual),
            "rotated_rest":rigid_failure_values(&rigid_failure_read_f32(device,reduction.staged.prot.handle(),3*rows)?),
            "vertex_body_u32":rigid_failure_read_u32(device,reduction.staged.vbody.handle(),rows)?,
            "recursive_residual_f32":rigid_failure_values(&rigid_failure_read_f32(device,handles.r,dim)?),
            "true_residual_l1_f64":true_residual.iter().map(|x|f64::from(x.abs())).sum::<f64>()}))
    })();
    // Buffer has no releasing Drop. Attempt EVERY free, including on partial
    // allocation/readback failure; retain the original diagnostic error.
    let mut cleanup_error=None;
    for buffer in [&mut probe,&mut out,&mut full,&mut applied,&mut absolute,&mut curvature] {
        if let Err(e)=buffer.free(device) {if cleanup_error.is_none(){cleanup_error=Some(e);}}
    }
    if let Some(fault)=fault_before {
        let bytes=std::slice::from_raw_parts(fault.as_ptr().cast::<u8>(),fault.len()*4);
        if let Err(e)=device.write(joints.fault,0,bytes){if cleanup_error.is_none(){cleanup_error=Some(e);}}
    }
    match result {Err(e)=>Err(e),Ok(value)=>if let Some(e)=cleanup_error{Err(e)}else{Ok(value)}}
}
fn rigid_failure_write(dir:&str,value:&serde_json::Value)->Result<(),String> {
    let path=std::path::Path::new(dir);
    if !path.is_absolute(){return Err("capture directory must be absolute".into());}
    std::fs::create_dir_all(path).map_err(|e|e.to_string())?;
    let path=path.join("rigid-pcg-failure.json");
    let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(&path).map_err(|e|e.to_string())?;
    use std::io::Write;
    file.write_all(&serde_json::to_vec_pretty(value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    eprintln!("RIGID_PCG_FAILURE_CAPTURE {}",path.display());Ok(())
}

// One opt-in, selected last-Newton witness. All GPU arithmetic below executes
// only after the original B0 cap is already true. No preconditioner/CSR dump.
#[derive(Default)]
pub(super) struct FrozenNewtonWitness {
    fit: Buffer<crate::data::PdrdRigidState>,
    reference: Buffer<crate::data::PdrdRigidState>,
    targets: Buffer<crate::joint_root_motion_gpu::RootTargetGpu>,
    force: Buffer<f32>, eval: Buffer<f32>, prot: Buffer<f32>,
    vbody: Buffer<u32>, cloth: Buffer<u32>,
    a: Buffer<f32>, b: Buffer<f32>, c: Buffer<f32>,
    rhs: Buffer<f32>, xr: Buffer<f32>, recursive: Buffer<f32>,
    op: Option<Operator<'static>>, joints: Option<super::joints::View>, roots: Option<super::roots::View>,
    tag: Option<serde_json::Value>, complete: bool, error: Option<String>,
}
impl FrozenNewtonWitness {
    fn copy<T:ppf_cts_compute::Pod>(device:&mut impl Device,source:ppf_cts_compute::Handle,
        target:&mut Buffer<T>,count:usize)->Result<(),Fault> {
        let bytes=count.checked_mul(std::mem::size_of::<T>()).ok_or_else(||Self::fault("copy extent overflow"))?;
        if (source.size as usize)<count {return Err(Self::fault("source extent too short"));}
        target.size(device,count,AllocLabel("newton.witness.frozen"))?;
        if bytes>0 {device.copy(target.handle(),0,source,0,bytes)?;}
        Ok(())
    }
    fn fault(message:&str)->Fault {Fault::Platform{call:"optional frozen Newton witness",detail:message.into()}}
    /// D2D only. Called after the exact B17 Newton solve, before later fit writes.
    /// # Safety
    /// Every supplied handle is the same-device live B17 handle and sized by
    /// its native producer. Qualification excludes cloth, hard/native locks and
    /// body projection, and admits exactly 16 vertices/2 bodies/12 DOFs/one root.
    /// Copies preserve the fitted/rotated-rest/operator generation used by solve.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn freeze<D:Device>(&mut self,device:&mut D,op:&Operator,
        reduction:&mut super::pdrd::Reduction<'_>,joints:super::joints::View,
        roots:Option<super::roots::View>,force:ppf_cts_compute::Handle,
        eval:ppf_cts_compute::Handle,rigid:&mut RigidWorkspace,tag:serde_json::Value) {
        self.complete=false;self.tag=Some(tag);self.error=None;
        if op.has_tet_factor(){self.error=Some("optional frozen witness needs owned tet factor snapshot".into());return;}
        let result=(||->Result<(),Fault>{
            let root=roots.ok_or_else(||Self::fault("root view missing"))?;
            if op.rows()!=16 || reduction.map.dim!=12 || reduction.map.n_bodies!=2
                || reduction.map.body_base!=0 || reduction.map.needs_projection()
                || joints.bodies!=2 || joints.base!=0 || joints.dim!=12
                || root.count!=1 || root.bodies!=2 || root.base!=0 || root.dim!=12
                || std::mem::size_of::<crate::data::PdrdRigidState>()!=104
                || std::mem::size_of::<crate::joint_root_motion_gpu::RootTargetGpu>()!=64 {
                return Err(Self::fault("outside bounded two-body witness domain"));
            }
            let rhs=rigid.f.handle();let xr=rigid.xr.handle();let recursive=rigid.r.handle();
            Self::copy(device,joints.state,&mut self.fit,2)?;
            Self::copy(device,joints.reference,&mut self.reference,2)?;
            Self::copy(device,root.targets,&mut self.targets,1)?;
            Self::copy(device,force,&mut self.force,48)?;
            Self::copy(device,eval,&mut self.eval,48)?;
            Self::copy(device,reduction.staged.prot.handle(),&mut self.prot,48)?;
            Self::copy(device,reduction.staged.vbody.handle(),&mut self.vbody,16)?;
            Self::copy(device,reduction.staged.cloth_off.handle(),&mut self.cloth,16)?;
            Self::copy(device,rhs,&mut self.rhs,12)?;
            Self::copy(device,xr,&mut self.xr,12)?;
            Self::copy(device,recursive,&mut self.recursive,12)?;
            if op.fixed.value.size>100_000 || op.dynamic.is_some_and(|d|d.value.size>100_000) {
                return Err(Self::fault("diagnostic matrix-value extent exceeds bounded scope"));
            }
            Self::copy(device,op.fixed.value,&mut self.b,op.fixed.value.size as usize)?;
            Self::copy(device,op.diagonal,&mut self.c,144)?;
            let dynamic=if let Some(d)=op.dynamic {
                Self::copy(device,d.value,&mut self.a,d.value.size as usize)?;
                Some(super::operator::DynamicView{value:self.a.handle(),..d})
            }else{None};
            self.op=Some(Operator{fixed_pattern: None, tet_factor:None,dynamic,fixed:super::spmv::FixedCsrView{
                index:op.fixed.index,offset:op.fixed.offset,value:self.b.handle(),
                transpose_pair:op.fixed.transpose_pair,transpose_offset:op.fixed.transpose_offset,rows:16},
                diagonal:self.c.handle()});
            // Record families are immutable scene uploads; reference/state/target
            // are copied. Original fault lane is NEVER a diagnostic destination.
            self.joints=Some(super::joints::View{state:self.fit.handle(),reference:self.reference.handle(),..joints});
            self.roots=Some(super::roots::View{targets:self.targets.handle(),..root});
            if let Some(t)=self.tag.as_mut().and_then(serde_json::Value::as_object_mut) {
                t.insert("joint_view".into(),serde_json::json!({"count":joints.count,"control_count":joints.control_count,
                    "extension_count":joints.extension_count,"drive_count":joints.drive_count,
                    "start_time_bits":joints.start_time_s.to_bits(),"sample_time_bits":joints.sample_time_s.to_bits(),
                    "dt_bits":joints.dt.to_bits(),"bodies":joints.bodies,"base":joints.base,"dim":joints.dim}));
                t.insert("source_handles".into(),serde_json::json!({
                    "fit":format!("{:?}",joints.state),"reference":format!("{:?}",joints.reference),
                    "root_targets":format!("{:?}",root.targets),"force":format!("{:?}",force),
                    "eval":format!("{:?}",eval),"prot":format!("{:?}",reduction.staged.prot.handle()),
                    "rhs":format!("{:?}",rhs),"xr":format!("{:?}",xr),"recursive_r":format!("{:?}",recursive),
                    "C":format!("{:?}",op.diagonal),"B_values":format!("{:?}",op.fixed.value),
                    "A_values":op.dynamic.map(|d|format!("{:?}",d.value)),
                    "ball":format!("{:?}",joints.ball),"angular":format!("{:?}",joints.angular),
                    "control":format!("{:?}",joints.control),"limits":format!("{:?}",joints.limits),
                    "drives":format!("{:?}",joints.drives),"device_keys":format!("{:?}",joints.device_keys),
                    "fixed_index":format!("{:?}",op.fixed.index),"fixed_offset":format!("{:?}",op.fixed.offset),
                    "fixed_transpose_pair":format!("{:?}",op.fixed.transpose_pair),"fixed_transpose_offset":format!("{:?}",op.fixed.transpose_offset),
                    "dynamic_metadata":op.dynamic.map(|d|serde_json::json!({"index":format!("{:?}",d.index),
                        "offset":format!("{:?}",d.offset),"transpose_index":format!("{:?}",d.transpose_index),
                        "transpose_offset":format!("{:?}",d.transpose_offset),"transpose_value":format!("{:?}",d.transpose_value)}))}));
            }
            Ok(())
        })();
        match result {Ok(())=>self.complete=true,Err(e)=>self.error=Some(e.to_string())}
    }
    pub(super) fn optional_panic(&mut self) {self.complete=false;self.error=Some("optional D2D freeze panicked".into());}
    /// The original cap has already fired. No physical buffer is an output.
    /// # Safety
    /// Freeze completed in the same attempt; retained CSR metadata and immutable
    /// joint record handles are still live. Physical paths B18..B33 cannot write
    /// those retained arrays before the next B0. The map remains unchanged.
    pub(super) unsafe fn at_cap<D:Device>(&mut self,device:&mut D,map:&super::rigid_map::RigidMap,
        bound_step:u32)->serde_json::Value {
        if !self.complete {return serde_json::json!({"status":"UNAVAILABLE_OR_INCOMPLETE_FROZEN_WITNESS",
            "tag":self.tag,"error":self.error,"no_terminal_substitution":true});}
        if self.tag.as_ref().and_then(|t|t["step"].as_u64()).map(|v|v+1)!=Some(u64::from(bound_step))
            || bound_step!=2048 || map.nrow!=16 || map.dim!=12 || map.body_base!=0
            || map.n_bodies!=2 || map.needs_projection() {
            return serde_json::json!({"status":"INVALID_WITNESS_BINDING","no_probe_executed":true});
        }
        let mut probe=Buffer::<f32>::default();let mut out=Buffer::<f32>::default();
        let mut full=Buffer::<f32>::default();let mut applied=Buffer::<f32>::default();
        let mut absolute=Buffer::<f32>::default();let mut curvature=Buffer::<f32>::default();
        let mut shadow_fault=Buffer::<u32>::default();
        let result=(||->Result<serde_json::Value,Fault>{
            let op=self.op.as_ref().ok_or_else(||Self::fault("frozen operator absent"))?;
            let source_joints=self.joints.ok_or_else(||Self::fault("frozen joint view absent"))?;
            let root=self.roots.ok_or_else(||Self::fault("frozen root view absent"))?;
            let fault_count=source_joints.control_count.max(source_joints.extension_count) as usize;
            let fault_before=if fault_count>0 {rigid_failure_read_u32(device,source_joints.fault,fault_count)?}else{Vec::new()};
            Self::copy(device,source_joints.fault,&mut shadow_fault,fault_count)?;
            let joints=super::joints::View{fault:shadow_fault.handle(),..source_joints};
            probe.size(device,12,AllocLabel("newton.witness.probe"))?;
            out.size(device,12,AllocLabel("newton.witness.output"))?;
            full.size(device,48,AllocLabel("newton.witness.full"))?;
            applied.size(device,48,AllocLabel("newton.witness.applied"))?;
            absolute.size(device,16,AllocLabel("newton.witness.absolute"))?;
            curvature.size(device,16,AllocLabel("newton.witness.curvature"))?;
            let vertex_body=rigid_failure_read_u32(device,self.vbody.handle(),16)?;
            let cloth_offset=rigid_failure_read_u32(device,self.cloth.handle(),16)?;
            if vertex_body!=map.vbody || cloth_offset!=map.cloth_off {return Err(Self::fault("frozen topology/map mismatch"));}
            // These are the exact original prolong/restrict entry records. Body
            // projection is a proven no-op in this qualified FREE-body domain.
            let prolong=|device:&mut D,input,output|->Result<(),Fault>{
                let args=super::kernels::PdrdProlongRowArgs{vertex_body:self.vbody.handle(),
                    cloth_offset:self.cloth.handle(),rotated_rest:self.prot.handle(),
                    reduced:input,full:output,body_base:0,count:16,seam_arena_count:0};
                device.launch("newton.witness.prolong",&args,16).map(|_|())
            };
            let restrict=|device:&mut D,input,output|->Result<(),Fault>{
                device.fill_zero(output,12*4)?;
                let args=super::kernels::PdrdRestrictRowArgs{vertex_body:self.vbody.handle(),
                    cloth_offset:self.cloth.handle(),rotated_rest:self.prot.handle(),full:input,
                    cloth_out:output,reduced:output,body_base:0,count:16,seam_arena_count:0};
                device.launch("newton.witness.restrict",&args,16).map(|_|())
            };
            // No input Q here: the root particular must survive the affine apply.
            let affine=|device:&mut D,input,output|->Result<(),Fault>{
                prolong(device,input,full.handle())?;
                op.apply(device,full.handle(),applied.handle(),absolute.handle(),curvature.handle())?;
                restrict(device,applied.handle(),output)?;
                joints.add(device,1,input,output)?;root.project(device,output)
            };
            let mut q_columns=Vec::new();let mut a_columns=Vec::new();
            for column in 0..12 {
                let mut unit=[0.0f32;12];unit[column]=1.0;probe.write(device,0,&unit)?;
                root.project(device,probe.handle())?;
                q_columns.push(rigid_failure_values(&rigid_failure_read_f32(device,probe.handle(),12)?));
                affine(device,probe.handle(),out.handle())?;
                a_columns.push(rigid_failure_values(&rigid_failure_read_f32(device,out.handle(),12)?));
            }
            restrict(device,self.force.handle(),out.handle())?;
            let restricted=rigid_failure_read_f32(device,out.handle(),12)?;
            joints.add(device,0,out.handle(),out.handle())?;
            root.project(device,out.handle())?;
            let rebuilt_rhs=rigid_failure_read_f32(device,out.handle(),12)?;
            device.fill_zero(probe.handle(),12*4)?;
            joints.add(device,0,probe.handle(),probe.handle())?;
            root.project(device,probe.handle())?;
            let isolated_joint_gradient=rigid_failure_read_f32(device,probe.handle(),12)?;
            affine(device,self.xr.handle(),out.handle())?;
            let affine_product=rigid_failure_read_f32(device,out.handle(),12)?;
            combine(device,self.rhs.handle(),out.handle(),probe.handle(),1.0,-1.0,12)?;
            root.project(device,probe.handle())?;
            let true_residual=rigid_failure_read_f32(device,probe.handle(),12)?;
            device.fill_zero(probe.handle(),12*4)?;
            root.particular(device,self.fit.handle(),probe.handle())?;
            let particular=rigid_failure_read_f32(device,probe.handle(),12)?;
            affine(device,probe.handle(),out.handle())?;
            let particular_product=rigid_failure_read_f32(device,out.handle(),12)?;
            let rhs=rigid_failure_read_f32(device,self.rhs.handle(),12)?;
            let xr=rigid_failure_read_f32(device,self.xr.handle(),12)?;
            let recursive=rigid_failure_read_f32(device,self.recursive.handle(),12)?;
            let fault_after=if fault_count>0 {rigid_failure_read_u32(device,source_joints.fault,fault_count)?}else{Vec::new()};
            let shadow=rigid_failure_read_u32(device,shadow_fault.handle(),fault_count)?;
            if fault_before!=fault_after {return Err(Self::fault("original joint fault lane changed"));}
            let mut fit_bytes=vec![0u8;208];device.read(self.fit.handle(),0,&mut fit_bytes)?;
            let mut ref_bytes=vec![0u8;208];device.read(self.reference.handle(),0,&mut ref_bytes)?;
            let mut root_bytes=vec![0u8;64];device.read(self.targets.handle(),0,&mut root_bytes)?;
            Ok(serde_json::json!({"schema":"frozen-newton-operator-witness-v1","status":"CAP_ONLY_FROZEN_DEVICE_WITNESS",
                "tag":self.tag,"history":false,"bound_iteration":bound_step,"last_completed_iteration":bound_step-1,
                "read_only_physical_state":true,"joint_fault_original_preserved":true,
                "joint_probe_geometry_ok":shadow.iter().all(|v|*v==0),
                "joint_fault_before_u32":fault_before,"joint_fault_after_u32":fault_after,"shadow_joint_fault_u32":shadow,
                "Q_columns":q_columns,"Q_A_Q_columns":a_columns,"matrix_layout":"array of device-applied column vectors",
                "full_force":rigid_failure_values(&rigid_failure_read_f32(device,self.force.handle(),48)?),
                "assembly_eval_xyz":rigid_failure_values(&rigid_failure_read_f32(device,self.eval.handle(),48)?),
                "rotated_rest":rigid_failure_values(&rigid_failure_read_f32(device,self.prot.handle(),48)?),
                "vertex_body_u32":vertex_body,"cloth_offset_u32":cloth_offset,
                "frozen_fit_104byte_records_u8":fit_bytes,"frozen_reference_104byte_records_u8":ref_bytes,
                "frozen_RootTargetGpu_64byte_records_u8":root_bytes,
                "stored_projected_rhs":rigid_failure_values(&rhs),"restricted_full_force_before_joint_Q":rigid_failure_values(&restricted),
                "rebuilt_projected_rhs":rigid_failure_values(&rebuilt_rhs),"isolated_projected_joint_gradient":rigid_failure_values(&isolated_joint_gradient),
                "actual_affine_xr":rigid_failure_values(&xr),"recursive_residual":rigid_failure_values(&recursive),
                "Q_A_actual_affine_xr":rigid_failure_values(&affine_product),"true_Q_f_minus_Ax":rigid_failure_values(&true_residual),
                "frozen_root_particular":rigid_failure_values(&particular),"Q_A_root_particular":rigid_failure_values(&particular_product),
                "full_C_column_major":rigid_failure_values(&rigid_failure_read_f32(device,self.c.handle(),144)?),
                "limits":"No CSR dump, no preconditioner inverse, no true-history claim. Atomic restriction replay may differ in low bits. Q_A_Q excludes affine root input; true residual uses full xr, never Qxr."}))
        })();
        let mut cleanup_error=None;
        for buf in [&mut probe,&mut out,&mut full,&mut applied,&mut absolute,&mut curvature] {
            if let Err(e)=buf.free(device) {if cleanup_error.is_none(){cleanup_error=Some(e.to_string());}}
        }
        if let Err(e)=shadow_fault.free(device) {if cleanup_error.is_none(){cleanup_error=Some(e.to_string());}}
        match result {
            Ok(mut value)=>{if let Some(object)=value.as_object_mut(){object.insert("diagnostic_cleanup_error".into(),serde_json::json!(cleanup_error));}value},
            Err(e)=>serde_json::json!({"status":"OPTIONAL_WITNESS_FAILED","tag":self.tag,"error":e.to_string(),"cleanup_error":cleanup_error,"original_rejection_retained":true})
        }
    }
    pub(super) fn cleanup(&mut self,device:&mut impl Device) {
        if self.tag.is_none(){return;}
        for b in [&mut self.force,&mut self.eval,&mut self.prot,&mut self.a,&mut self.b,&mut self.c,
            &mut self.rhs,&mut self.xr,&mut self.recursive] {let _=b.free(device);}
        let _=self.fit.free(device);let _=self.reference.free(device);let _=self.targets.free(device);
        let _=self.vbody.free(device);let _=self.cloth.free(device);
        self.complete=false;
    }
}

#[cfg(all(test, feature="cpu"))]
#[path="warm_alloc_fault_test.rs"]
mod warm_alloc_fault_test;

#[cfg(all(test, feature="cpu"))]
#[path="zero_solve_export_test.rs"]
mod zero_solve_export_test;
fn sparse_host_body_tree(device: &impl Device, host_mode: bool) -> bool {
    host_mode && device.info().backend == "host"
        && std::env::var_os("PPF_BODY_TREE_HOST_SPARSE").is_some_and(|value| value == "1")
}

fn body_tree_scalar_extent(bodies: usize, factor_mode: bool, sparse_host: bool) -> Option<usize> {
    let dofs = bodies.checked_mul(6)?;
    if dofs == 0 || dofs > u32::MAX as usize {
        return None;
    }
    if sparse_host {
        // A host forest stores 6x6 factors per body, not a dense DOF square.
        let factor_bytes = bodies.checked_mul(std::mem::size_of::<[[f64; 6]; 6]>())?;
        if factor_bytes > isize::MAX as usize {
            return None;
        }
    } else if !factor_mode && dofs > 1024 {
        return None;
    }
    Some(dofs)
}

#[cfg(test)]
mod sparse_host_extent_tests {
    #[test]
    fn a_sparse_chain_solves_above_the_dense_dof_limit() {
        use super::super::{body_block_projection::SparseBodyMatrix, shadow_tree::FreeForest};

        // Just above the old dense limit, with no split of the component.
        let bodies = 1024 / 6 + 1;
        assert_eq!(super::body_tree_scalar_extent(bodies, false, false), None);
        assert_eq!(super::body_tree_scalar_extent(bodies, false, true), Some(6 * bodies));
        assert_eq!(super::body_tree_scalar_extent(usize::MAX, false, true), None);
        let diagonal = |value| {
            let mut block = [[0.0f32; 6]; 6];
            for (i, row) in block.iter_mut().enumerate() {
                row[i] = value;
            }
            block
        };
        let mut blocks = std::collections::BTreeMap::new();
        for body in 0..bodies {
            blocks.insert((body, body), diagonal(4.0));
            if body > 0 {
                blocks.insert((body - 1, body), diagonal(-1.0));
                blocks.insert((body, body - 1), diagonal(-1.0));
            }
        }
        let matrix = SparseBodyMatrix { body_count: bodies, free: (0..bodies).collect(), blocks };
        let forest = FreeForest::from_sparse_weighted(&matrix, bodies).expect("sparse SPD chain");
        assert_eq!(forest.omitted_cycles, 0);
        let rhs: Vec<_> = (0..6 * bodies).map(|lane| (lane % 13) as f64 - 6.0).collect();
        let solution = forest.apply(&rhs).expect("finite sparse solve");
        for (lane, &expected) in rhs.iter().enumerate() {
            let mut applied = 4.0 * solution[lane];
            if lane >= 6 { applied -= solution[lane - 6]; }
            if lane + 6 < solution.len() { applied -= solution[lane + 6]; }
            assert!((applied - expected).abs() < 1e-11, "lane {lane}");
        }
    }
}

unsafe fn prepare_body_tree<D:Device>(device:&mut D,op:&Operator,reduction:&mut super::pdrd::Reduction<'_>,joints:super::joints::View,roots:Option<super::roots::View>,host_mode:bool,factor_mode:bool)->super::scene::FatalResult<Option<(PreparedBodyTree,Option<BodyTreeDebug>,Option<BodyGroupReference>)>> {
    let bodies=reduction.map.n_bodies;
    let sparse_host = sparse_host_body_tree(device, host_mode);
    let root_count = roots.map_or(0, |root| root.count as usize);
    if sparse_host && super::body_block_capture::host_sparse_plan(op, joints, bodies, 0, root_count).is_none() {
        eprintln!("PPF_BODY_TREE_PCG active=false reason=host_sparse_memory_budget");
        return Ok(None);
    }
    let mut fixed=Vec::new();
    if fixed.try_reserve_exact(bodies).is_err() {
        eprintln!("PPF_BODY_TREE_PCG active=false reason=host_sparse_workspace");
        return Ok(None);
    }
    fixed.resize(bodies, false);
    if let Some(root)=roots {let mut targets=vec![crate::joint_root_motion_gpu::RootTargetGpu::default();root.count as usize];let bytes=std::slice::from_raw_parts_mut(targets.as_mut_ptr().cast::<u8>(),targets.len()*std::mem::size_of::<crate::joint_root_motion_gpu::RootTargetGpu>());if !bytes.is_empty(){device.read(root.targets,0,bytes)?;}for r in targets{if r.enabled!=0{let b=r.body as usize;if b>=bodies{eprintln!("PPF_BODY_TREE_PCG active=false reason=root_mapping");return Ok(None)}fixed[b]=true;}}}
    let free_count = fixed.iter().filter(|&&is_fixed| !is_fixed).count();
    let mut free = Vec::new();
    let mut roots = Vec::new();
    if free.try_reserve_exact(free_count).is_err()
        || roots.try_reserve_exact(bodies - free_count).is_err()
    {
        eprintln!("PPF_BODY_TREE_PCG active=false reason=body_tree_workspace");
        return Ok(None);
    }
    for (body, is_fixed) in fixed.iter().copied().enumerate() {
        if is_fixed { roots.push(body); } else { free.push(body); }
    }
    let Some(m) = body_tree_scalar_extent(free.len(), factor_mode, sparse_host) else {
        eprintln!("PPF_BODY_TREE_PCG active=false reason=free_extent");
        return Ok(None);
    };
    let plan = if sparse_host {
        let Some(plan) = super::body_block_capture::host_sparse_plan(op, joints, bodies, free.len(), root_count) else {
            eprintln!("PPF_BODY_TREE_PCG active=false reason=host_sparse_memory_budget");
            return Ok(None);
        };
        Some(plan)
    } else {
        None
    };
    let mut slots = Vec::new();
    if slots.try_reserve_exact(m).is_err() {
        eprintln!("PPF_BODY_TREE_PCG active=false reason=host_sparse_workspace");
        return Ok(None);
    }
    slots.extend(free.iter().flat_map(|&b|(0..6).map(move |i|6*b+i)));
    let t=std::time::Instant::now();
    if factor_mode || sparse_host {
        let sparse=match super::body_block_capture::capture_sparse(device,op,reduction,joints,&slots,plan){
            Ok(x)=>x,
            Err(Fault::Platform{call:"direct body projection",detail})=>{eprintln!("PPF_BODY_TREE_PCG active=false reason=direct_{}",detail);return Ok(None)},
            Err(Fault::Alloc { .. }) if sparse_host => {
                eprintln!("PPF_BODY_TREE_PCG active=false reason=host_sparse_allocation");
                return Ok(None);
            }
            Err(e)=>return Err(e.into())
        };
        let host_timing = sparse_host.then(|| (t.elapsed().as_secs_f64() * 1000.0, std::time::Instant::now()));
        // The host uses the ordinary forest solve and has no CUDA packet
        // limit. Keep the whole weighted forest that the dense host path
        // would choose from the same blocks.
        let capacity = if sparse_host { free.len() } else { 170 };
        let tree=match super::shadow_tree::FreeForest::from_sparse_weighted(&sparse,capacity){Ok(x)=>x,Err(e)=>{eprintln!("PPF_BODY_TREE_PCG active=false reason=spd_{:?}",e.0);return Ok(None)}};
        if let Some((direct_ms, factor_start)) = host_timing {
            let factor_ms = factor_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("PPF_BODY_TREE_PREP apply=host_sparse direct_ms={direct_ms:.6} factor_ms={factor_ms:.6}");
            return Ok(Some((PreparedBodyTree::Host(HostBodyTree::new(tree,direct_ms,factor_ms)),None,None)));
        }
        let (view,free_map,body_count)=tree.factor_view();
        let packed=match super::body_factor_pack::PackedFactor::from_view(view,free_map,body_count){Ok(x)=>x,Err(e)=>{eprintln!("PPF_BODY_TREE_PCG active=false reason=factor_pack_{:?}",e);return Ok(None)}};
        let owner=super::body_factor_device::BodyFactorDevice::upload(device,&packed)?;
        return Ok(Some((PreparedBodyTree::Factor(owner),None,None)));
    }
    let columns=match super::body_block_capture::capture(device,op,reduction,joints,&slots){Ok(x)=>x,Err(Fault::Platform{call:"direct body projection",detail})=>{eprintln!("PPF_BODY_TREE_PCG active=false reason=direct_{}",detail);return Ok(None)},Err(e)=>return Err(e.into())};let direct_ms=t.elapsed().as_secs_f64()*1000.;
    let mut diag=vec![[[0f64;6];6];bodies];for a in &mut diag {for i in 0..6 {a[i][i]=1.;}}
    let t=std::time::Instant::now();let seed=match super::shadow_tree::prepare(&diag,&fixed,&[]){Ok(x)=>x,Err(e)=>{eprintln!("PPF_BODY_TREE_PCG active=false reason=seed_{:?}",e.0);return Ok(None)}};
    let tree=match seed.from_dense_weighted(&columns){Ok(x)=>x,Err(e)=>{eprintln!("PPF_BODY_TREE_PCG active=false reason=spd_{:?}",e.0);return Ok(None)}};let factor_ms=t.elapsed().as_secs_f64()*1000.;
    if factor_mode {
        let (view,free_map,body_count)=tree.factor_view();
        let packed=match super::body_factor_pack::PackedFactor::from_view(view,free_map,body_count) {
            Ok(value)=>value,
            Err(e)=>{eprintln!("PPF_BODY_TREE_PCG active=false reason=factor_pack_{:?}",e);return Ok(None)}
        };
        let owner=super::body_factor_device::BodyFactorDevice::upload(device,&packed)?;
        return Ok(Some((PreparedBodyTree::Factor(owner),None,None)));
    }
    if host_mode {
        eprintln!("PPF_BODY_TREE_PREP apply=host direct_ms={:.6} factor_ms={:.6}",direct_ms,factor_ms);
        return Ok(Some((PreparedBodyTree::Host(HostBodyTree::new(tree,direct_ms,factor_ms)),None,None)));
    }
    let t=std::time::Instant::now();let mut matrix=vec![0f64;m*m];let mut basis=vec![0f64;6*bodies];for (j,&slot) in slots.iter().enumerate(){basis[slot]=1.;let result=tree.apply(&basis);basis[slot]=0.;let x=match result {Ok(x)=>x,Err(e)=>{eprintln!("PPF_BODY_TREE_PCG active=false reason=inverse_{:?}",e.0);return Ok(None)}};for (i,&r) in slots.iter().enumerate(){matrix[i*m+j]=x[r];}}
    let packed=match super::body_inverse_pack::pack(&matrix,&free,&roots,bodies){Ok(p)=>p,Err(e)=>{eprintln!("PPF_BODY_TREE_PCG active=false reason=pack_{}",e);return Ok(None)}};let inverse_pack_ms=t.elapsed().as_secs_f64()*1000.;let t=std::time::Instant::now();let owner=super::body_inverse_device::BodyInverseDevice::upload(device,&packed)?;
    eprintln!("PPF_BODY_TREE_PREP direct_ms={:.6} factor_ms={:.6} inverse_pack_ms={:.6} upload_ms={:.6}",direct_ms,factor_ms,inverse_pack_ms,t.elapsed().as_secs_f64()*1000.);let reference=if std::env::var_os("PPF_BODY_GROUP_SOLVE").is_some(){Some(BodyGroupReference{columns,free,roots})}else{None};let debug=if std::env::var_os("PPF_BODY_TREE_DEBUG_RESTART").is_some_and(|v|v=="1"){Some(BodyTreeDebug{tree,packed})}else{None};Ok(Some((PreparedBodyTree::Device(owner),debug,reference)))
}
struct BodyTreeDebug {tree:super::shadow_tree::FreeForest,packed:super::body_inverse_pack::Packed}
unsafe fn debug_body_apply<D:Device>(d:&mut D,owner:&super::body_inverse_device::BodyInverseDevice,debug:&BodyTreeDebug,reduction:&mut super::pdrd::Reduction<'_>,factor:ppf_cts_compute::Handle,inverse:ppf_cts_compute::Handle,roots:Option<super::roots::View>,r:ppf_cts_compute::Handle,body_r:ppf_cts_compute::Handle,base:usize,bodies:usize)->Result<(),Fault>{
    let n=6*bodies;let mut tmp=Buffer::<f32>::default();
    let result=(||->Result<(),Fault>{tmp.size(d,base+n,ppf_cts_compute::AllocLabel("body_tree.debug"))?;
        reduction.apply_precond(d,factor,inverse,r,tmp.handle())?;
        if let Some(root)=roots {root.project(d,tmp.handle())?;}
        let before=rigid_failure_read_f32(d,tmp.handle(),base)?;
        owner.apply(d,body_r,tmp.span(base,n))?;
        if let Some(root)=roots {root.project(d,tmp.handle())?;}
        let allr=rigid_failure_read_f32(d,r,base+n)?;let allz=rigid_failure_read_f32(d,tmp.handle(),base+n)?;
        let gpu=allz[base..].to_vec();let x=&allr[base..];let rhs:Vec<f64>=x.iter().map(|&v|v as f64).collect();
        let reference=debug.tree.apply(&rhs).map_err(|e|Fault::Platform{call:"body tree debug",detail:e.0})?;
        let p=&debug.packed;let mut f32sum=vec![0f32;n];let mut f64sum=vec![0f64;n];
        for row in 0..p.rows as usize {for k in p.offset[row] as usize..p.offset[row+1] as usize {let c=p.index[k] as usize;for i in 0..3 {let mut block=0f32;for j in 0..3 {let a=p.value[9*k+3*j+i];block+=a*x[3*c+j];f64sum[3*row+i]+=(a as f64)*(x[3*c+j] as f64);}f32sum[3*row+i]+=block;}}
            for k in p.transpose_offset[row] as usize..p.transpose_offset[row+1] as usize {let c=p.transpose_pair[2*k] as usize;let slot=p.transpose_pair[2*k+1] as usize;for i in 0..3 {let mut block=0f32;for j in 0..3 {let a=p.value[9*slot+3*i+j];block+=a*x[3*c+j];f64sum[3*row+i]+=(a as f64)*(x[3*c+j] as f64);}f32sum[3*row+i]+=block;}}}
        let rel=|a:&[f64],b:&[f64]|{let mut s=0.;let mut q=0.;for (&x,&y) in a.iter().zip(b){s+=(x-y)*(x-y);q+=y*y;}if q>0.{(s/q).sqrt()}else{s.sqrt()}};
        let gf:Vec<f64>=gpu.iter().map(|&v|v as f64).collect();let hf:Vec<f64>=f32sum.iter().map(|&v|v as f64).collect();
        let dot=|v:&[f64]|rhs.iter().zip(v).map(|(a,b)|a*b).sum::<f64>();
        let max_abs=gf.iter().zip(&reference).map(|(a,b)|(a-b).abs()).fold(0f64,f64::max);let finite=gf.iter().chain(&reference).chain(&hf).chain(&f64sum).all(|v|v.is_finite());
        let fem_r=allr[..base].iter().map(|v|(*v as f64).powi(2)).sum::<f64>().sqrt();let fem_z=allz[..base].iter().map(|v|(*v as f64).powi(2)).sum::<f64>().sqrt();let fem_delta=allz[..base].iter().zip(&before).map(|(a,b)|((*a as f64)-(*b as f64)).powi(2)).sum::<f64>().sqrt();
        eprintln!("PPF_BODY_TREE_RESTART_DEBUG finite={} gpu_tree_max_abs={:.9e} gpu_vs_host32={:.9e} host32_vs_pack64={:.9e} pack64_vs_tree64={:.9e} gpu_vs_tree64={:.9e} rdot_gpu={:.9e} rdot_tree={:.9e} fem_r_norm={:.9e} fem_z_norm={:.9e} fem_delta={:.9e}",finite,max_abs,rel(&gf,&hf),rel(&hf,&f64sum),rel(&f64sum,&reference),rel(&gf,&reference),dot(&gf),dot(&reference),fem_r,fem_z,fem_delta);Ok(())})();
    let cleanup=tmp.free(d);match result{Err(e)=>Err(e),Ok(())=>cleanup}
}
struct ScalarDebug {solve:usize,rz:f64,reads:usize,bytes:usize,read_ms:f64,sum_ms:f64,x0norm:f64}
impl ScalarDebug {
 fn dot(&mut self,d:&mut impl Device,a:ppf_cts_compute::Handle,b:ppf_cts_compute::Handle,n:usize)->Result<f64,Fault>{
  let t=std::time::Instant::now();let x=rigid_failure_read_f32(d,a,n)?;let y=rigid_failure_read_f32(d,b,n)?;self.read_ms+=t.elapsed().as_secs_f64()*1000.;self.reads+=2;self.bytes+=8*n;
  let t=std::time::Instant::now();let result=rigid_failure_dot64(&x,&y);self.sum_ms+=t.elapsed().as_secs_f64()*1000.;if !result.is_finite(){return Err(Fault::Platform{call:"scalar64 diagnostic",detail:"nonfinite dot; invalid diagnostic".into()});}Ok(result)
 }
 fn receipt(&self,iteration:u32,recursive:f32,verified:f32,reason:&str){eprintln!("PPF_BODY_SCALAR64 solve={} reason={} iteration={} recursive={:.9e} true={:.9e} x0_norm={:.9e} reads={} bytes={} read_ms={:.6} host_sum_ms={:.6}",self.solve,reason,iteration,recursive,verified,self.x0norm,self.reads,self.bytes,self.read_ms,self.sum_ms);}
}
struct BodyGroupReference {columns:Vec<f32>,free:Vec<usize>,roots:Vec<usize>}
fn body_group_reference(d:&mut impl Device,reference:&BodyGroupReference,xh:ppf_cts_compute::Handle,fh:ppf_cts_compute::Handle,ah:ppf_cts_compute::Handle,rh:ppf_cts_compute::Handle,base:usize,bodies:usize,initial:&[f32],err0:f32,joints:super::joints::View)->Result<(),Fault>{
 fn bad(s:&str)->Fault{Fault::Platform{call:"body group reference",detail:s.into()}}
 let m=bodies.checked_mul(6).ok_or_else(||bad("extent overflow"))?;
 if base!=0||m==0||m>1024||!reference.roots.is_empty()||reference.free.len()!=bodies||reference.free.iter().enumerate().any(|(i,&b)|i!=b)||reference.columns.len()!=m*m||initial.len()!=bodies+1 {eprintln!("PPF_BODY_GROUP_REFERENCE unsupported=domain");return Ok(())}
 let x=rigid_failure_read_f32(d,xh,m)?;let f=rigid_failure_read_f32(d,fh,m)?;let a=rigid_failure_read_f32(d,ah,m)?;let r=rigid_failure_read_f32(d,rh,m)?;
 if reference.columns.iter().chain(&x).chain(&f).chain(&a).chain(&r).chain(initial).any(|v|!v.is_finite())||!err0.is_finite(){return Err(bad("nonfinite input"));}
 let mut ax=vec![0f64;m];let mut rr=vec![0f64;m];for i in 0..m{for j in 0..m{ax[i]+=reference.columns[j*m+i] as f64*x[j] as f64;}rr[i]=f[i] as f64-ax[i];}
 if ax.iter().chain(&rr).any(|v|!v.is_finite()){return Err(bad("nonfinite result"));}
 let mut groups=vec![0f64;bodies+1];for b in 0..bodies{groups[b+1]=rr[6*b..6*b+6].iter().map(|v|v.abs()).sum();}
 let mut worst=1usize;let mut ratio=0f64;for b in 0..bodies{let i=b+1;let start=initial[i] as f64;let scale=if start>0.{start}else{err0 as f64};if scale<=0.{return Err(bad("invalid group scale"));}let native=r[6*b..6*b+6].iter().map(|v|v.abs() as f64).sum::<f64>()/scale;if native>ratio{ratio=native;worst=i;}}
 let k=6*(worst-1);eprintln!("PPF_BODY_GROUP_REFERENCE layout=raw_column_major worst_group={} native_ratio={:.9e} ref_groups={:?} body={} f={:?} x={:?} gpu_ax={:?} gpu_r={:?} ref_ax={:?} ref_r={:?}",worst,ratio,groups,worst-1,&f[k..k+6],&x[k..k+6],&a[k..k+6],&r[k..k+6],&ax[k..k+6],&rr[k..k+6]);
 if let Some(path)=std::env::var_os("PPF_BODY_GROUP_SNAPSHOT"){
  use std::io::Write;
  let identity=joints.shadow_identity.map(|(u,s)|serde_json::json!({"scene_uuid":String::from_utf8_lossy(&u),"scene_seal":String::from_utf8_lossy(&s)}));
  let doc=serde_json::json!({"schema":"ppf.body-group-reference.v1","layout":"raw column-major B[j*m+i]","body_base":base,"bodies":bodies,"free_bodies":reference.free,"roots":reference.roots,"start_time_s":joints.start_time_s,"sample_time_s":joints.sample_time_s,"dt":joints.dt,"identity":identity,"initial":initial,"err0":err0,"columns":reference.columns,"x":x,"f":f,"native_ax":a,"native_r":r,"reference_ax_f64":ax,"reference_r_f64":rr,"reference_group_l1":groups});
  let bytes=serde_json::to_vec(&doc).map_err(|e|bad(&e.to_string()))?;if bytes.len()>32*1024*1024{return Err(bad("snapshot exceeds 32MiB"));}
  let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(path).map_err(|e|bad(&e.to_string()))?;file.write_all(&bytes).map_err(|e|bad(&e.to_string()))?;file.flush().map_err(|e|bad(&e.to_string()))?;
 }
 Ok(())
}
enum PreparedBodyTree {Factor(super::body_factor_device::BodyFactorDevice),Device(super::body_inverse_device::BodyInverseDevice),Host(HostBodyTree)}
struct HostBodyTree {tree:super::shadow_tree::FreeForest,direct_ms:f64,factor_ms:f64,reads:u64,bytes:u64,read_ms:f64,solve_ms:f64,write_ms:f64}
impl HostBodyTree {
 fn new(tree:super::shadow_tree::FreeForest,direct_ms:f64,factor_ms:f64)->Self{Self{tree,direct_ms,factor_ms,reads:0,bytes:0,read_ms:0.,solve_ms:0.,write_ms:0.}}
 fn apply(&mut self,d:&mut impl Device,r:ppf_cts_compute::Handle,z:ppf_cts_compute::Handle,base:usize,bodies:usize)->Result<(),Fault>{
  let (read,solve,write,bytes)=host_tree_apply(&self.tree,d,r,z,base,bodies)?;self.read_ms+=read;self.solve_ms+=solve;self.write_ms+=write;self.reads+=1;self.bytes+=bytes;Ok(())
 }
 fn receipt(&self){eprintln!("PPF_BODY_TREE_HOST_RECEIPT direct_ms={:.6} factor_ms={:.6} calls={} transfer_bytes={} read_ms={:.6} solve_ms={:.6} write_ms={:.6}",self.direct_ms,self.factor_ms,self.reads,self.bytes,self.read_ms,self.solve_ms,self.write_ms);}
}

fn host_tree_apply(tree:&super::shadow_tree::FreeForest,d:&mut impl Device,r:ppf_cts_compute::Handle,z:ppf_cts_compute::Handle,base:usize,bodies:usize)->Result<(f64,f64,f64,u64),Fault>{
 let n=bodies.checked_mul(6).ok_or_else(||Fault::Shape{kernel:"body.tree.host",detail:"body extent overflow".into()})?;
 let end=base.checked_add(n).ok_or_else(||Fault::Shape{kernel:"body.tree.host",detail:"body extent overflow".into()})?;
 let offset=base.checked_mul(4).ok_or_else(||Fault::Shape{kernel:"body.tree.host",detail:"body offset overflow".into()})?;
 if end>r.size as usize || end>z.size as usize{return Err(Fault::Shape{kernel:"body.tree.host",detail:"body span mismatch".into()})}
 let t=std::time::Instant::now();let mut input=vec![0f32;n];let bytes=unsafe{std::slice::from_raw_parts_mut(input.as_mut_ptr().cast::<u8>(),4*n)};d.read(r,offset,bytes)?;let read=t.elapsed().as_secs_f64()*1000.;
 let t=std::time::Instant::now();let rhs:Vec<f64>=input.iter().map(|&v|v as f64).collect();let x=tree.apply(&rhs).map_err(|e|Fault::Platform{call:"body tree host solve",detail:e.0})?;let mut result=Vec::with_capacity(n);for v in x{let q=v as f32;if !q.is_finite(){return Err(Fault::Platform{call:"body tree host solve",detail:"nonfinite f32 result".into()});}result.push(q);}let solve=t.elapsed().as_secs_f64()*1000.;
 let t=std::time::Instant::now();let bytes=unsafe{std::slice::from_raw_parts(result.as_ptr().cast::<u8>(),4*n)};d.write(z,offset,bytes)?;let write=t.elapsed().as_secs_f64()*1000.;Ok((read,solve,write,8*n as u64))
}

fn pack_shadow_tree(tree:&super::shadow_tree::FreeForest,bodies:usize)->Result<super::body_inverse_pack::Packed,String>{
 let slots=tree.free_indices();let m=slots.len();if m==0||m>1024{return Err("shadow inverse extent".into())}let mut free=Vec::new();for c in slots.chunks_exact(6){if c.iter().enumerate().any(|(i,&v)|v!=c[0]+i)||c[0]%6!=0{return Err("shadow free mapping".into())}free.push(c[0]/6);}
 let roots:Vec<_>=(0..bodies).filter(|b|!free.contains(b)).collect();let mut a=vec![0f64;m*m];let mut unit=vec![0f64;6*bodies];for (j,&v) in slots.iter().enumerate(){unit[v]=1.;let x=tree.apply(&unit).map_err(|e|e.0)?;unit[v]=0.;for (i,&r) in slots.iter().enumerate(){a[i*m+j]=x[r];}}
 super::body_inverse_pack::pack(&a,&free,&roots,bodies).map_err(str::to_owned)
}
