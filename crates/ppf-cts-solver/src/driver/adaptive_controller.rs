//! Versioned host outer-loop lifecycle, shared by production and host tests.
//! Only authored dt is restored; native-written time/prev_dt are never copied.
use super::adaptive_retry::RetryRun;

pub const POLICY: &str = "sticky-ceiling-v2";

pub trait DtParameter {
    fn authored_dt(&self) -> f32;
    fn set_dt(&mut self, dt: f32);
}

pub use crate::checkpoint_values::Controller;

#[derive(Clone, Copy)]
pub struct Selection {
    pub logical_step: u64,
    pub authored_nominal_h: f32,
    pub controller_ceiling_h: f32,
    pub initial_trial_h: f32,
}

struct RestoreAuthored<'a, P: DtParameter> {
    param: &'a mut P,
    authored: f32,
}
impl<P: DtParameter> Drop for RestoreAuthored<'_, P> {
    fn drop(&mut self) { self.param.set_dt(self.authored); }
}

/// Production owns the real schedule sampler and full attempt preparation;
/// this helper owns their ordering, authored-dt restoration and sticky state.
/// Decreases clip the ceiling; increases keep the last safe ceiling. No growth.
/// Disabled mode samples and runs once without changing/restoring any field.
pub fn logical_step<P, S, R>(
    enabled: bool,
    preserve_authored_on_host_retry: bool,
    controller: &mut Controller,
    param: &mut P,
    start: f64,
    configuration: &str,
    cg_limit: u32,
    cg_tolerance: f32,
    sample: S,
    run: R,
) -> RetryRun
where P: DtParameter, S: FnOnce(&mut P), R: FnOnce(&mut P, Selection) -> RetryRun,
{
    if !enabled {
        sample(param);
        let nominal = param.authored_dt();
        if preserve_authored_on_host_retry {
            let restore=RestoreAuthored{param,authored:nominal};
            return run(&mut *restore.param,Selection{logical_step:controller.accepted,authored_nominal_h:nominal,controller_ceiling_h:nominal,initial_trial_h:nominal});
        }
        return run(param, Selection { logical_step: controller.accepted, authored_nominal_h: nominal, controller_ceiling_h: nominal, initial_trial_h: nominal });
    }
    let mut restore = RestoreAuthored { param, authored: controller.authored };
    let previous_ceiling=controller.ceiling;
    restore.param.set_dt(controller.authored);
    sample(restore.param);
    let authored = restore.param.authored_dt();
    assert!(authored.is_finite() && authored > 0.0 && start.is_finite(), "CG_ADAPTIVE_V2_INVALID_AUTHORED_OR_START");
    restore.authored = authored;
    controller.authored = authored;
    controller.ceiling = controller.ceiling.min(authored);
    let selection = Selection { logical_step: controller.accepted, authored_nominal_h: authored, controller_ceiling_h: controller.ceiling, initial_trial_h: authored.min(controller.ceiling) };
    // Direct stdout is deliberate: these opt-in evidence records must survive
    // RUST_LOG=warn. println's write error panics rather than fabricating output.
    println!("CG_ADAPTIVE_V2_BEGIN: policy={} config={} logical_step={} start={:.17e} authored_nominal_h={:.9e} controller_ceiling_h={:.9e} initial_trial_h={:.9e} cg_limit={} cg_tolerance={:.9e} rejection_budget=16", POLICY, configuration, selection.logical_step, start, authored, selection.controller_ceiling_h, selection.initial_trial_h, cg_limit, cg_tolerance);
    let outcome = run(restore.param, selection);
    for event in &outcome.events {
        println!("CG_ADAPTIVE_V2_RETRY: policy={} config={} logical_step={} start={:.17e} attempt={} requested_h={:.9e} iterations={} residual={:.9e} native_rejections={} next_h={:.9e} remaining_after={}", POLICY, configuration, selection.logical_step, start, event.attempt_index, event.requested_h, event.iterations, event.residual, event.native_rejections, event.next_h, event.remaining_budget);
    }
    if outcome.save_requested {
        controller.ceiling=previous_ceiling;
    } else if outcome.result.success() {
        let advance = outcome.result.time - start;
        assert!(advance.is_finite() && advance > 0.0, "CG_ADAPTIVE_V2_COMMIT_INVALID");
        let remaining = outcome.final_remaining_budget.unwrap_or_else(||outcome.events.last().map_or(16, |event| event.remaining_budget).checked_sub(outcome.result.retry_rejections).expect("CG_ADAPTIVE_V2_ACCEPTED_BUDGET_INVALID"));
        controller.ceiling = outcome.cg_ceiling_candidate.unwrap_or(outcome.requested_h).min(authored);
        controller.accepted = controller.accepted.checked_add(1).expect("adaptive accepted counter overflow");
        println!("CG_ADAPTIVE_V2_ACCEPTED: policy={} config={} logical_step={} start={:.17e} accepted_count={} attempt={} requested_h={:.9e} actual_advance={:.17e} endpoint={:.17e} next_ceiling_h={:.9e} native_rejections={} remaining_after={}", POLICY, configuration, selection.logical_step, start, controller.accepted, outcome.retry_count, outcome.requested_h, advance, outcome.result.time, controller.ceiling, outcome.result.retry_rejections, remaining);
    } else {
        if preserve_authored_on_host_retry {controller.ceiling=previous_ceiling;}
        println!("CG_ADAPTIVE_V2_REFUSED: policy={} config={} logical_step={} start={:.17e} attempt={} requested_h={:.9e} retry_reason={} iterations={} residual={:.9e} native_rejections={} ceiling_unchanged_h={:.9e}", POLICY, configuration, selection.logical_step, start, outcome.retry_count, outcome.requested_h, outcome.result.retry_reason, outcome.result.retry_iterations, outcome.result.retry_residual, outcome.result.retry_rejections, controller.ceiling);
    }
    // Drop restores authored dt on success, refusal and unwinding, while
    // preserving the actual accepted clock, previous dt and all other fields.
    drop(restore);
    outcome
}
