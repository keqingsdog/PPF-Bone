//! A bounded trial of the complete frozen rigid body graph.
use super::*;
use ppf_cts_compute::Handle;

const MAX_BYTES: usize = 64 << 20;
const MAX_FILL: usize = 16_384;
const MAX_UPDATES: usize = 131_072;
const MAX_CHECKS: u32 = 2;
const MAX_BORROWS: u8 = 4;
const FAILURE_COOLDOWN: u8 = 3;
// Require a worst ratio at most one quarter on the original group scales.
const MAX_BORROWED_REFINE_RATIO: f32 = 0.25;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) struct ReuseDomain {
    workspace: u64,
    binding: u64,
    dt_bits: u32,
    roots: Option<super::super::roots::ConstraintDomain>,
}

impl ReuseDomain {
    pub(super) fn new(
        workspace: u64,
        binding: u64,
        dt: f32,
        roots: Option<super::super::roots::View>,
    ) -> Option<Self> {
        if workspace == 0
            || binding == 0
            || !dt.is_finite()
            || dt <= 0.0
            || roots.is_some_and(|root| !root.constraint_domain.is_cacheable())
        {
            return None;
        }
        Some(Self {
            workspace,
            binding,
            dt_bits: dt.to_bits(),
            roots: roots.map(|root| root.constraint_domain),
        })
    }
}

#[derive(Default)]
pub(super) struct Reuse {
    entry: Option<Cached>,
    cooldown: Option<(ReuseDomain, u8)>,
}

struct Cached {
    domain: ReuseDomain,
    budget: usize,
    free: Vec<usize>,
    borrows: u8,
    factor: full_body_factor::Factor,
}

impl Cached {
    fn usable(&self, domain: ReuseDomain, free: &[usize], budget: usize) -> bool {
        self.domain == domain
            && self.free == free
            && self.budget == budget
            && self.borrows < MAX_BORROWS
    }
}

pub(super) struct Accepted {
    pub x: Vec<f32>,
    pub score: f32,
    pub checks: u32,
}

#[derive(Default)]
struct PrivateJoints {
    buffers: Vec<(Handle, bool, Buffer<u32>)>,
}

impl PrivateJoints {
    fn bind(
        &mut self,
        device: &mut impl Device,
        joints: super::super::joints::View,
    ) -> Result<super::super::joints::View, Fault> {
        joints.with_preview_buffers(&mut |source, fault| {
            if source.size == 0 {
                return Ok(source);
            }
            if let Some((_, _, buffer)) = self.buffers.iter().find(|(handle, kind, _)| {
                handle.arena == source.arena
                    && handle.off == source.off
                    && handle.size == source.size
                    && *kind == fault
            }) {
                return Ok(buffer.handle());
            }
            self.buffers.push((source, fault, Buffer::none()));
            let buffer = &mut self.buffers.last_mut().expect("private joint buffer").2;
            let count = source.size as usize;
            buffer.size(device, count, AllocLabel("pcg.full_body.joints"))?;
            if fault {
                device.fill_zero(buffer.handle(), count * 4)?;
            } else {
                device.copy(buffer.handle(), 0, source, 0, count * 4)?;
            }
            Ok(buffer.handle())
        })
    }

    fn free(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        let mut first = Ok(());
        for (_, _, mut buffer) in self.buffers.drain(..) {
            let freed = buffer.free(device);
            first = first.and(freed);
        }
        first
    }
}

fn faults_clear(
    device: &mut impl Device,
    joints: super::super::joints::View,
) -> Result<bool, Fault> {
    for handle in joints.shadow_fault_handles() {
        let bytes = handle.size as usize * 4;
        // One bounded transfer covers the common full joint fault bank.
        let mut chunk = [0u8; 16 * 1024];
        for start in (0..bytes).step_by(chunk.len()) {
            let size = chunk.len().min(bytes - start);
            device.read(handle, start, &mut chunk[..size])?;
            if chunk[..size].iter().any(|byte| *byte != 0) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[derive(Default)]
struct Trial {
    x: Buffer<f32>,
    r: Buffer<f32>,
    ap: Buffer<f32>,
    xv: Buffer<f32>,
    mxv: Buffer<f32>,
    row_absolute: Buffer<f32>,
    row_curvature: Buffer<f32>,
    norms: ReadbackBuffer<f32>,
}

impl Trial {
    fn size(
        &mut self,
        device: &mut impl Device,
        dim: usize,
        rows: usize,
        bodies: usize,
    ) -> Result<(), Fault> {
        let label = AllocLabel("pcg.full_body.trial");
        self.x.size(device, dim, label)?;
        self.r.size(device, dim, label)?;
        self.ap.size(device, dim, label)?;
        self.xv.size(device, 3 * rows, label)?;
        self.mxv.size(device, 3 * rows, label)?;
        self.row_absolute.size(device, rows, label)?;
        self.row_curvature.size(device, rows, label)?;
        self.norms.size(device, 1 + bodies, label)?;
        Ok(())
    }

    fn free(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        let mut first = self.norms.free(device);
        for buffer in [
            &mut self.x,
            &mut self.r,
            &mut self.ap,
            &mut self.xv,
            &mut self.mxv,
            &mut self.row_absolute,
            &mut self.row_curvature,
        ] {
            let freed = buffer.free(device);
            first = first.and(freed);
        }
        first
    }
}

fn reserve<T>(count: usize) -> Option<Vec<T>> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(count).ok()?;
    Some(buffer)
}

fn limit_bytes() -> usize {
    std::env::var("PPF_BODY_FULL_NATIVE_BUDGET_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map_or(MAX_BYTES, |value| value.min(MAX_BYTES))
}

fn grouped_score(groups: &[f32], initial: &[f32], err0: f32) -> Option<f32> {
    if groups
        .iter()
        .any(|score| !score.is_finite() || *score < 0.0)
    {
        return None;
    }
    let score = super::super::pdrd::worst_relative_residual(groups, initial, err0);
    score.is_finite().then_some(score)
}

fn read_vector(device: &mut impl Device, handle: Handle, out: &mut [f32]) -> Result<(), Fault> {
    let byte_count = std::mem::size_of_val(out);
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), byte_count) };
    device.read(handle, 0, bytes)
}

pub(super) unsafe fn try_solve<D: Device>(
    device: &mut D,
    op: &Operator<'_>,
    reduction: &mut super::super::pdrd::Reduction<'_>,
    joints: super::super::joints::View,
    roots: Option<super::super::roots::View>,
    rigid: &RigidWorkspace,
    seed_residual: Handle,
    work: &Workspace,
    err0: f32,
    tolerance: f32,
    coalesced_reference: bool,
    trace: bool,
    mut reuse: Option<(&mut Reuse, ReuseDomain)>,
    mut symbolic: Option<&mut full_body_factor::SymbolicCache>,
) -> super::super::scene::FatalResult<Option<Accepted>> {
    use super::super::{body_block_capture, phase};
    let bodies = reduction.map.n_bodies;
    let dim = reduction.map.dim;
    let budget = limit_bytes();
    macro_rules! refuse {
        ($reason:expr) => {{
            let reason = $reason;
            if trace {
                eprintln!("PPF_BODY_FULL_NATIVE active=false reason={reason}");
            }
            return Ok(None);
        }};
    }
    if !err0.is_finite()
        || err0 <= 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || rigid
            .group_initial
            .iter()
            .any(|score| !score.is_finite() || *score < 0.0)
    {
        refuse!("invalid_reference");
    }
    let root_count = roots.map_or(0, |root| root.count as usize);
    // Admission is required before allocating even the root map or reading the device.
    if budget / 4096 < bodies
        || body_block_capture::host_sparse_plan_at_limit(op, joints, bodies, 0, root_count, budget)
            .is_none()
    {
        refuse!("capture_budget");
    }
    let Some(mut fixed) = reserve(bodies) else {
        refuse!("workspace_allocation")
    };
    fixed.resize(bodies, false);
    if let Some(root) = roots {
        use crate::joint_root_motion_gpu::RootTargetGpu;
        let Some(mut targets) = reserve::<RootTargetGpu>(root_count) else {
            refuse!("workspace_allocation")
        };
        targets.resize(root_count, RootTargetGpu::default());
        let byte_count = std::mem::size_of_val(targets.as_slice());
        let bytes = std::slice::from_raw_parts_mut(targets.as_mut_ptr().cast::<u8>(), byte_count);
        if !bytes.is_empty() {
            device.read(root.targets, 0, bytes)?;
        }
        for target in targets.into_iter().filter(|target| target.enabled != 0) {
            let Some(body) = fixed.get_mut(target.body as usize) else {
                refuse!("root_layout")
            };
            *body = true;
        }
    }
    let free_count = fixed.iter().filter(|&&is_fixed| !is_fixed).count();
    if free_count == 0 {
        refuse!("empty_free_domain");
    }
    let Some(plan) = body_block_capture::host_sparse_plan_at_limit(
        op, joints, bodies, free_count, root_count, budget,
    ) else {
        refuse!("capture_budget")
    };
    let Some(mut free) = reserve(free_count) else {
        refuse!("workspace_allocation")
    };
    let Some(mut slots) = reserve(free_count * 6) else {
        refuse!("workspace_allocation")
    };
    for (body, fixed) in fixed.into_iter().enumerate() {
        if !fixed {
            free.push(body);
            slots.extend(6 * body..6 * body + 6);
        }
    }
    if !faults_clear(device, joints)? {
        refuse!("existing_native_fault");
    }

    let mut attempt = |device: &mut D,
                       borrowed: Option<&full_body_factor::Factor>,
                       refine_borrowed: bool|
     -> super::super::scene::FatalResult<
        Option<(Accepted, Option<full_body_factor::Factor>)>,
    > {
        let mut private = PrivateJoints::default();
        let mut trial = Trial::default();
        let attempted = (|| -> super::super::scene::FatalResult<_> {
            let joints = match private.bind(device, joints) {
                Ok(value) => value,
                Err(Fault::Alloc { .. }) => refuse!("optional_joint_scratch_allocation"),
                Err(error) => return Err(error.into()),
            };
            let (fresh, capture_ms, factor_ms, directed) = if borrowed.is_none() {
                let capture_started = std::time::Instant::now();
                let matrix = {
                    let _phase = phase::start("pcg.full_body.capture");
                    match body_block_capture::capture_sparse_quiet(
                        device, op, reduction, joints, &slots, plan,
                    ) {
                        Ok(matrix) => matrix,
                        Err(Fault::Platform {
                            call:
                                "direct body projection"
                                | "legacy reduced Schwarz frozen export"
                                | "reduced Schwarz frozen joint export",
                            detail,
                        }) => refuse!(detail),
                        Err(error) => return Err(error.into()),
                    }
                };
                let capture_ms = 1000.0 * capture_started.elapsed().as_secs_f64();
                if !faults_clear(device, joints)? {
                    refuse!("capture_numeric_fault");
                }
                let directed = matrix.blocks.len();
                let factor_started = std::time::Instant::now();
                let factor = {
                    let _phase = phase::start("pcg.full_body.factor");
                    let budget = full_body_factor::Budget {
                        bytes: budget,
                        fill_blocks: MAX_FILL,
                    };
                    let built = if let Some(cache) = symbolic.as_deref_mut() {
                        full_body_factor::Factor::build_compact(
                            free_count, &matrix.blocks, budget, MAX_UPDATES, cache,
                        ).map(|(factor, hit)| {
                            if trace {
                                eprintln!("PPF_BODY_FULL_FACTOR_COMPACT symbolic_hit={hit}");
                            }
                            factor
                        })
                    } else {
                        full_body_factor::Factor::build_compute_bounded(
                            free_count, &matrix.blocks, budget, MAX_UPDATES,
                        )
                    };
                    match built {
                        Ok(factor) => factor,
                        Err(reason) => refuse!(reason),
                    }
                };
                let factor_ms = 1000.0 * factor_started.elapsed().as_secs_f64();
                drop(matrix);
                (Some(factor), capture_ms, factor_ms, directed)
            } else {
                (None, 0.0, 0.0, 0)
            };
            let factor = borrowed
                .or(fresh.as_ref())
                .expect("borrowed or freshly factored");
            let Some(mut x) = reserve(dim) else {
                refuse!("workspace_allocation")
            };
            let Some(mut r) = reserve(dim) else {
                refuse!("workspace_allocation")
            };
            let Some(mut rhs) = reserve(free_count) else {
                refuse!("workspace_allocation")
            };
            let Some(mut groups) = reserve(1 + bodies) else {
                refuse!("workspace_allocation")
            };
            x.resize(dim, 0.0);
            r.resize(dim, 0.0);
            groups.resize(1 + bodies, 0.0);
            read_vector(device, rigid.xr.handle(), &mut x)?;
            read_vector(device, seed_residual, &mut r)?;
            match trial.size(device, dim, op.rows() as usize, bodies) {
                Ok(()) => {}
                Err(Fault::Alloc { .. }) => refuse!("optional_trial_scratch_allocation"),
                Err(error) => return Err(error.into()),
            }

            let max_checks = if borrowed.is_some() && !refine_borrowed {
                1
            } else {
                MAX_CHECKS
            };
            for check in 1..=max_checks {
                {
                    let _phase = phase::start("pcg.full_body.substitution");
                    rhs.clear();
                    rhs.extend(
                        free.iter()
                            .map(|&body| std::array::from_fn(|lane| f64::from(r[6 * body + lane]))),
                    );
                    let correction = match factor.solve(&rhs) {
                        Ok(delta) => delta,
                        Err(reason) => refuse!(reason),
                    };
                    for (&body, delta) in free.iter().zip(correction) {
                        for (lane, delta) in delta.into_iter().enumerate() {
                            x[6 * body + lane] = (f64::from(x[6 * body + lane]) + delta) as f32;
                        }
                    }
                }
                if x.iter().any(|value| !value.is_finite()) {
                    refuse!("nonfinite_trial");
                }
                {
                    let _phase = phase::start("pcg.full_body.true_residual");
                    trial.x.write(device, 0, &x)?;
                    let (input, product, residual) =
                        (trial.x.handle(), trial.ap.handle(), trial.r.handle());
                    let (xv, mxv) = (trial.xv.handle(), trial.mxv.handle());
                    let (absolute, curvature) =
                        (trial.row_absolute.handle(), trial.row_curvature.handle());
                    reduction.project_bodies(device, input)?;
                    if let Some(root) = roots {
                        root.project(device, input)?;
                        root.particular(device, joints.state, input)?;
                    }
                    if coalesced_reference {
                        device.run("pcg.rigid.reference_native", |encoder| {
                            reduction.encode_prolong(encoder, input, xv)?;
                            op.encode_apply(encoder, xv, mxv, absolute, curvature)?;
                            reduction.encode_restrict(encoder, mxv, product, true)
                        })?;
                    } else {
                        reduction.prolong(device, input, xv)?;
                        op.apply(device, xv, mxv, absolute, curvature)?;
                        reduction.restrict(device, mxv, product, &rigid.zero)?;
                    }
                    // Action 1 is the affine reference used by the ordinary final check.
                    joints.add(device, 1, input, product)?;
                    reduction.project_bodies(device, product)?;
                    if let Some(root) = roots {
                        root.project(device, product)?;
                    }
                    if coalesced_reference {
                        group_l1_after(
                            device,
                            residual,
                            &mut trial.norms,
                            &work.fold,
                            0,
                            bodies,
                            &mut groups,
                            |encoder| {
                                encode_combine_raw(
                                    encoder,
                                    rigid.f.handle(),
                                    product,
                                    residual,
                                    1.0,
                                    -1.0,
                                    dim,
                                )?;
                                if let Some(root) = roots {
                                    root.encode_project(encoder, residual)?;
                                }
                                Ok(())
                            },
                        )?;
                    } else {
                        combine(device, rigid.f.handle(), product, residual, 1.0, -1.0, dim)?;
                        reduction.project_bodies(device, residual)?;
                        if let Some(root) = roots {
                            root.project(device, residual)?;
                        }
                        group_l1_device(
                            device,
                            residual,
                            &mut trial.norms,
                            &work.fold,
                            0,
                            bodies,
                            &mut groups,
                        )?;
                    }
                    if !faults_clear(device, joints)? {
                        refuse!("trial_numeric_fault");
                    }
                    let score = grouped_score(&groups, &rigid.group_initial, err0);
                    if let Some(score) = score.filter(|&score| score < tolerance) {
                        read_vector(device, input, &mut x)?;
                        if x.iter().any(|value| !value.is_finite()) {
                            refuse!("nonfinite_canonical_trial");
                        }
                        if trace && borrowed.is_none() {
                            eprintln!("PPF_BODY_FULL_NATIVE active=true checks={check} bodies={bodies} free={free_count} directed={directed} capture_ms={capture_ms:.6} factor_ms={factor_ms:.6} true_relative={score:.9e} tolerance={tolerance:.9e} stats={:?}", factor.stats);
                        }
                        return Ok(Some((
                            Accepted {
                                x,
                                score,
                                checks: check,
                            },
                            fresh,
                        )));
                    }
                    let refine = check < max_checks
                        && (borrowed.is_none()
                            || score.is_some_and(|score| score <= MAX_BORROWED_REFINE_RATIO));
                    if borrowed.is_some() && trace {
                        eprintln!("PPF_BODY_FULL_REUSE event=verify_rejected check={check} true_relative={:.9e} tolerance={tolerance:.9e} refine={refine}",
                            score.unwrap_or(f32::INFINITY));
                    }
                    if !refine {
                        break;
                    }
                    if groups.iter().any(|score| !score.is_finite()) {
                        refuse!("nonfinite_true_residual");
                    }
                    read_vector(device, residual, &mut r)?;
                    read_vector(device, input, &mut x)?;
                }
            }
            refuse!("true_residual")
        })();
        let trial_cleanup = trial.free(device);
        let joint_cleanup = private.free(device);
        let accepted = attempted?;
        trial_cleanup?;
        joint_cleanup?;
        Ok(accepted)
    };

    if let Some((state, domain)) = reuse.as_mut() {
        let mut skip = false;
        if let Some((previous_domain, left)) = state.cooldown.as_mut() {
            if previous_domain == domain && *left > 0 {
                *left -= 1;
                skip = true;
            } else {
                state.cooldown = None;
            }
        }
        if let Some(mut previous) = state.entry.take() {
            if !skip && previous.usable(*domain, &free, budget) {
                let _phase = phase::start("pcg.full_body.reuse_try");
                let started = std::time::Instant::now();
                let force_reject = std::env::var_os("PPF_BODY_FULL_REUSE_FORCE_REJECT")
                    .is_some_and(|value| value == "1");
                let handles = [
                    rigid.xr.handle(),
                    seed_residual,
                    rigid.p.handle(),
                    rigid.rp.handle(),
                    rigid.f.handle(),
                ];
                let prior = if force_reject {
                    Some(attempt_buffers::snapshot(device, handles, dim)?)
                } else {
                    None
                };
                let refine = !force_reject
                    && std::env::var_os("PPF_BODY_FULL_REUSE_REFINE")
                        .is_some_and(|value| value == "1");
                let result = attempt(device, Some(&previous.factor), refine)?;
                if let Some(prior) = prior {
                    if !attempt_buffers::unchanged(device, handles, dim, &prior)?
                        || !faults_clear(device, joints)?
                    {
                        return Err(super::super::scene::Fatal::invariant(
                            "full-body reuse changed live state before fresh fallback",
                        ));
                    }
                    eprintln!("PPF_BODY_FULL_REUSE_AUDIT seed_and_faults_unchanged=true");
                }
                let verified = result.is_some();
                if let Some((accepted, _)) = result {
                    if !force_reject {
                        previous.borrows += 1;
                        if trace {
                            eprintln!("PPF_BODY_FULL_REUSE event=accepted age={} checks={} true_relative={:.9e} tolerance={tolerance:.9e} elapsed_ms={:.6}", previous.borrows, accepted.checks, accepted.score, started.elapsed().as_secs_f64() * 1000.0);
                        }
                        state.entry = Some(previous);
                        return Ok(Some(accepted));
                    }
                }
                state.cooldown = Some((*domain, FAILURE_COOLDOWN));
                if trace {
                    eprintln!(
                        "PPF_BODY_FULL_REUSE event=rejected forced={force_reject} verified={verified} elapsed_ms={:.6}",
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                }
            } else if trace {
                eprintln!(
                    "PPF_BODY_FULL_REUSE event=fresh reason={}",
                    if skip { "cooldown" } else { "domain_or_age" }
                );
            }
        }
    }

    let Some((accepted, fresh)) = attempt(device, None, false)? else {
        return Ok(None);
    };
    if let (Some((state, domain)), Some(factor)) = (reuse, fresh) {
        state.entry = Some(Cached {
            domain,
            budget,
            free,
            borrows: 0,
            factor,
        });
    }
    Ok(Some(accepted))
}

mod attempt_buffers {
    use super::*;

    pub(super) fn snapshot(
        device: &mut impl Device,
        handles: [Handle; 5],
        dim: usize,
    ) -> Result<Vec<Vec<f32>>, Fault> {
        handles
            .into_iter()
            .map(|handle| rigid_failure_read_f32(device, handle, dim))
            .collect()
    }

    pub(super) fn unchanged(
        device: &mut impl Device,
        handles: [Handle; 5],
        dim: usize,
        before: &[Vec<f32>],
    ) -> Result<bool, Fault> {
        for (handle, before) in handles.into_iter().zip(before) {
            let after = rigid_failure_read_f32(device, handle, dim)?;
            if !before
                .iter()
                .zip(after)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{full_body_factor, grouped_score, Cached, ReuseDomain, MAX_BORROWS};

    #[test]
    fn stale_domain_layout_limit_or_age_cannot_borrow() {
        let domain = ReuseDomain::new(1, 2, 0.005, None).unwrap();
        let diagonal =
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 2.0 } else { 0.0 }));
        let budget = full_body_factor::Budget {
            bytes: 1 << 20,
            fill_blocks: 0,
        };
        let factor =
            full_body_factor::Factor::build(1, &[((0, 0), diagonal)].into_iter().collect(), budget)
                .unwrap();
        let mut cache = Cached {
            domain,
            budget: budget.bytes,
            free: vec![8],
            borrows: 0,
            factor,
        };
        assert!(cache.usable(domain, &[8], budget.bytes));
        assert!(!cache.usable(domain, &[9], budget.bytes));
        assert!(!cache.usable(domain, &[8], 1));
        assert!(!cache.usable(
            ReuseDomain::new(2, 2, 0.005, None).unwrap(),
            &[8],
            budget.bytes
        ));
        assert!(!cache.usable(
            ReuseDomain::new(1, 3, 0.005, None).unwrap(),
            &[8],
            budget.bytes
        ));
        let changed_dt = f32::from_bits(0.005f32.to_bits() + 1);
        assert!(!cache.usable(
            ReuseDomain::new(1, 2, changed_dt, None).unwrap(),
            &[8],
            budget.bytes
        ));
        cache.borrows = MAX_BORROWS;
        assert!(!cache.usable(domain, &[8], budget.bytes));
        assert!(ReuseDomain::new(1, 0, 0.005, None).is_none());
    }

    #[test]
    fn one_bad_group_refuses_even_when_the_other_groups_are_small() {
        let initial = [0., 1024., 1.];
        let accept = |groups: &[f32], initial, err0, tolerance| {
            grouped_score(groups, initial, err0).filter(|&score| score < tolerance)
        };
        assert_eq!(accept(&[0., 1., 0.02], &initial, 1025., 0.01), None);
        assert_eq!(accept(&[0., 1., 0.01], &initial, 1025., 0.01), None);
        assert_eq!(accept(&[0., 1., f32::NAN], &initial, 1025., 0.01), None);
        assert_eq!(accept(&[0., 1., 0.004], &initial, 1025., 0.01), Some(0.004));
    }
}
