//! Optional stride-one policy for a compatible borrowed device factor.

#[derive(Clone, Copy)]
enum Mode {
    Frozen,
    Progress,
}

const WINDOW: u32 = 4;
const EXTENSION: u32 = 16;
const FALLBACK_ROOM: u32 = 4;
const FROZEN_EXTENSION: u32 = 64;
const FROZEN_STALL: u32 = 12;

pub(super) struct Trial {
    mode: Mode,
    trace: bool,
    history: [f32; WINDOW as usize],
    first_deferred: Option<u32>,
    best_residual: f32,
    last_progress: u32,
}

impl Trial {
    pub(super) fn configured() -> Option<Self> {
        use std::sync::OnceLock;
        static CONFIG: OnceLock<Option<(Mode, bool)>> = OnceLock::new();
        let (mode, trace) = *CONFIG.get_or_init(|| {
            let mode = match std::env::var("PPF_JOINT_FACTOR_REFRESH_POLICY").ok().as_deref() {
                Some("frozen") => Mode::Frozen,
                Some("progress") => Mode::Progress,
                _ => return None,
            };
            let trace = std::env::var_os("PPF_JOINT_FACTOR_REFRESH_RECEIPT").is_some_and(|v| v == "1");
            Some((mode, trace))
        }).as_ref()?;
        Some(Self::new(mode, trace))
    }

    fn new(mode: Mode, trace: bool) -> Self {
        Self { mode, trace, history: [f32::NAN; WINDOW as usize], first_deferred: None,
            best_residual: f32::INFINITY, last_progress: 0 }
    }

    /// The residual estimates whether it is economical to keep the old factor.
    /// This never accepts a solve or changes its tolerance or iteration cap.
    pub(super) fn defer(&mut self, iteration: u32, residual: f32, tolerance: f32,
                        cap: u32, original_due: bool) -> bool {
        let slot = (iteration % WINDOW) as usize;
        let previous = std::mem::replace(&mut self.history[slot], residual);
        let finite = residual.is_finite() && residual > 0.0;
        if finite && residual <= 0.95 * self.best_residual {
            self.best_residual = residual;
            self.last_progress = iteration;
        }
        if !original_due {
            return false;
        }
        let first = self.first_deferred.unwrap_or(iteration);
        let refresh_room = cap.saturating_sub(FALLBACK_ROOM);
        let defer = match self.mode {
            Mode::Frozen => finite && iteration < refresh_room
                && iteration < first.saturating_add(FROZEN_EXTENSION)
                && iteration.saturating_sub(self.last_progress) < FROZEN_STALL,
            Mode::Progress => {
                let end = first.saturating_add(EXTENSION).min(refresh_room);
                if !finite || !previous.is_finite() || residual > 0.8 * previous || iteration >= end
                    || !tolerance.is_finite() || tolerance <= 0.0 {
                    false
                } else {
                    let contraction = f64::from(residual) / f64::from(previous);
                    let target = f64::from(tolerance) / f64::from(residual);
                    let remaining = f64::from(WINDOW) * target.ln() / contraction.ln();
                    remaining.is_finite() && remaining >= 0.0 && remaining <= f64::from(end - iteration)
                }
            }
        };
        if self.trace {
            let policy = match self.mode { Mode::Frozen => "frozen", Mode::Progress => "progress" };
            if defer && self.first_deferred.is_none() {
                eprintln!("PPF_JOINT_FACTOR_REFRESH_TRIAL policy={policy} status=deferred iteration={iteration} residual={residual:.9e} previous={previous:.9e}");
            } else if !defer {
                let deferred = self.first_deferred.map_or(0, |first| iteration.saturating_sub(first));
                eprintln!("PPF_JOINT_FACTOR_REFRESH_TRIAL policy={policy} status=refresh iteration={iteration} residual={residual:.9e} deferred_iterations={deferred}");
            }
        }
        if defer {
            self.first_deferred.get_or_insert(iteration);
        }
        defer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn does_not_defer_without_a_progress_window_or_iteration_room() {
        let mut trial = Trial::new(Mode::Progress, false);
        assert!(!trial.defer(1, 1.0, 0.01, 100, true));
        assert!(!trial.defer(5, 0.02, 0.01, 9, true));
    }

    #[test]
    fn pays_for_a_short_trending_tail_then_refreshes_when_it_stalls() {
        let mut trial = Trial::new(Mode::Progress, false);
        for (index, residual) in [1.0, 0.75, 0.5, 0.25].into_iter().enumerate() {
            assert!(!trial.defer(index as u32 + 1, residual, 0.01, 100, false));
        }
        assert!(trial.defer(5, 0.15, 0.01, 100, true));
        assert!(trial.defer(6, 0.1, 0.01, 100, true));
        assert!(!trial.defer(7, 0.42, 0.01, 100, true));
    }

    #[test]
    fn extension_is_limited_even_with_a_good_recent_window() {
        let mut trial = Trial::new(Mode::Progress, false);
        trial.first_deferred = Some(5);
        trial.defer(17, 0.5, 0.01, 100, false);
        assert!(!trial.defer(21, 0.011, 0.01, 100, true));
    }

    #[test]
    fn frozen_probe_still_refreshes_invalid_or_stalled_solves() {
        let mut trial = Trial::new(Mode::Frozen, false);
        assert!(!trial.defer(1, 1.0, 0.01, 30, false));
        assert!(trial.defer(2, 2.0, 0.01, 30, true));
        assert!(!trial.defer(3, f32::NAN, 0.01, 30, true));
        assert!(!trial.defer(13, 1.0, 0.01, 30, true));
    }
}
