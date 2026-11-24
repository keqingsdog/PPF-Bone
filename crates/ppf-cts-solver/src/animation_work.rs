//! Per-accepted-step work accounting and the opt-in animation h controller.
//! No additional constraints, stopping rules, or state cross the native ABI.

use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Work {
    pub newton: u64,
    pub pcg: u64,
    pub caps: u32,
}

thread_local! {
    // The backend prepares and advances all candidates on its own thread.
    // Multiple native retries accumulate in the same accepted-step scope.
    static ACTIVE: Cell<Option<Work>> = const { Cell::new(None) };
}

pub(crate) struct WorkScope;

impl WorkScope {
    pub(crate) fn begin() -> Self {
        ACTIVE.set(Some(Work::default()));
        Self
    }

    pub(crate) fn finish(self) -> Work {
        ACTIVE.take().unwrap_or_default()
    }
}

impl Drop for WorkScope {
    fn drop(&mut self) {
        ACTIVE.set(None);
    }
}

pub(crate) fn measuring() -> bool {
    ACTIVE.get().is_some()
}

pub(crate) fn newton() {
    if let Some(mut count) = ACTIVE.get() {
        count.newton = count.newton.saturating_add(1);
        ACTIVE.set(Some(count));
    }
}

pub(crate) fn pcg(iterations: u32, iteration_cap: bool) {
    if let Some(mut count) = ACTIVE.get() {
        count.pcg = count.pcg.saturating_add(u64::from(iterations));
        count.caps = count.caps.saturating_add(u32::from(iteration_cap));
        ACTIVE.set(Some(count));
    }
}

#[derive(Clone, Copy)]
struct Rates {
    newton: f64,
    linear: f64,
}

impl Rates {
    fn worse_than(self, other: Self, factor: f64) -> bool {
        self.newton > factor * other.newton || self.linear > factor * other.linear
    }

    fn worth_promoting(self, other: Self) -> bool {
        !self.worse_than(other, 1.05)
            && (self.newton < 0.9 * other.newton || self.linear < 0.9 * other.linear)
    }
}

#[derive(Default)]
struct Window {
    steps: u32,
    seconds: f64,
    newton: f64,
    linear: f64,
}

impl Window {
    fn push(&mut self, seconds: f64, work: Work) {
        self.steps += 1;
        self.seconds += seconds;
        self.newton += work.newton as f64;
        // Charge a per-report baseline for a zero-iteration solve as well.
        self.linear += work.pcg.saturating_add(work.newton) as f64;
    }

    fn rates(&self) -> Rates {
        Rates { newton: self.newton / self.seconds, linear: self.linear / self.seconds }
    }
}

struct Probe {
    h: f32,
    reference: Rates,
    measured: Window,
}

pub(crate) struct Policy {
    authored_bits: Option<u32>,
    h: f32,
    window: Window,
    author_rate: Option<Rates>,
    stable_rate: Option<Rates>,
    probe: Option<Probe>,
    cooldown: u8,
    slower_steps: u8,
    next_larger: bool,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            authored_bits: None, h: 0.0, window: Window::default(), author_rate: None,
            stable_rate: None, probe: None, cooldown: 0, slower_steps: 0, next_larger: true,
        }
    }
}

impl Policy {
    pub(crate) fn choose(&mut self, authored: f32, fps: f64, authored_ceiling: bool) -> f32 {
        let frame = (1.0 / fps) as f32;
        if !authored.is_finite() || authored < f32::MIN_POSITIVE || !frame.is_finite() || frame <= 0.0 {
            return authored;
        }
        let limit = if authored_ceiling { authored } else { frame.max(authored) };
        if self.authored_bits != Some(authored.to_bits()) {
            *self = Self { authored_bits: Some(authored.to_bits()), h: authored, ..Self::default() };
        }
        if let Some(probe) = &self.probe {
            return probe.h.min(limit);
        }
        if self.window.steps < 4 || self.cooldown != 0 {
            return self.h.min(limit);
        }
        let reference = self.window.rates();
        let lower = (authored * 0.5).max(f32::MIN_POSITIVE);
        let larger = (self.h * 1.25).min(limit);
        let smaller = (self.h * 0.8).max(lower);
        let mut trial = if self.next_larger { larger } else { smaller };
        self.next_larger = !self.next_larger;
        if trial == self.h {
            trial = if self.next_larger { larger } else { smaller };
        }
        self.window = Window::default();
        if trial == self.h {
            return self.h;
        }
        self.probe = Some(Probe { h: trial, reference, measured: Window::default() });
        trial
    }

    pub(crate) fn observe(&mut self, authored: f32, seconds: f64, work: Work, rejections: u32) {
        if !seconds.is_finite() || seconds <= 0.0 || work.newton == 0 {
            return;
        }
        if work.caps != 0 || rejections != 0 {
            self.return_to_authored(authored);
            return;
        }
        if let Some(mut probe) = self.probe.take() {
            probe.measured.push(seconds, work);
            let rate = probe.measured.rates();
            if rate.worse_than(probe.reference, 1.5) {
                self.cooldown = 12;
            } else if probe.measured.steps < 2 {
                self.probe = Some(probe);
            } else if rate.worth_promoting(probe.reference)
                && self.author_rate.is_none_or(|author| !rate.worse_than(author, 1.05))
            {
                self.h = probe.h;
                self.stable_rate = Some(rate);
                self.cooldown = 4;
            } else {
                self.cooldown = 12;
            }
            return;
        }
        self.cooldown = self.cooldown.saturating_sub(1);
        self.window.push(seconds, work);
        let rate = self.window.rates();
        if self.h == authored {
            if self.window.steps >= 4 {
                self.author_rate = Some(rate);
            }
        } else if self.stable_rate.is_some_and(|reference| rate.worse_than(reference, 1.5)) {
            self.slower_steps += 1;
            if self.slower_steps >= 2 {
                self.return_to_authored(authored);
            }
        } else {
            self.slower_steps = 0;
        }
        // Retain a short baseline window rather than mixing distant motion
        // regimes across the entire simulation.
        if self.cooldown != 0 && self.window.steps >= 4 {
            self.window = Window::default();
        }
    }

    fn return_to_authored(&mut self, authored: f32) {
        self.h = authored;
        self.probe = None;
        self.window = Window::default();
        self.cooldown = 12;
        self.slower_steps = 0;
        self.stable_rate = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const NOMINAL: f32 = 0.01;
    const FPS: f64 = 24.0;
    const WORK: Work = Work { newton: 2, pcg: 10, caps: 0 };

    fn calibrated() -> Policy {
        let mut p = Policy::default();
        for _ in 0..4 {
            assert_eq!(p.choose(NOMINAL, FPS, false), NOMINAL);
            p.observe(NOMINAL, 0.01, WORK, 0);
        }
        p
    }

    #[test]
    fn benefits_in_both_rates_keep_the_probe() {
        let mut p = calibrated();
        let trial = p.choose(NOMINAL, FPS, false);
        assert!(trial > NOMINAL);
        p.observe(NOMINAL, 0.0125, WORK, 0);
        assert_eq!(p.choose(NOMINAL, FPS, false), trial);
        p.observe(NOMINAL, 0.0125, WORK, 0);
        assert_eq!(p.choose(NOMINAL, FPS, false), trial);
    }

    #[test]
    fn expensive_linear_rate_rejects_a_popular_time_step() {
        let mut p = calibrated();
        assert!(p.choose(NOMINAL, FPS, false) > NOMINAL);
        p.observe(NOMINAL, 0.0125, Work { pcg: 100, ..WORK }, 0);
        for _ in 0..4 {
            assert_eq!(p.choose(NOMINAL, FPS, false), NOMINAL);
            p.observe(NOMINAL, 0.01, WORK, 0);
        }
    }

    #[test]
    fn counts_all_native_attempts_and_hard_cap_cools_down() {
        let mut p = calibrated();
        assert!(p.choose(NOMINAL, FPS, false) > NOMINAL);
        let scope = WorkScope::begin();
        newton(); pcg(10_000, true);
        newton(); pcg(20, false);
        let work = scope.finish();
        assert_eq!((work.newton, work.pcg, work.caps), (2, 10_020, 1));
        assert!(!measuring());
        p.observe(NOMINAL, 0.006, work, 1);
        assert_eq!(p.choose(NOMINAL, FPS, false), NOMINAL);
    }

    #[test]
    fn an_authored_ceiling_still_permits_a_small_probe() {
        let mut p = calibrated();
        assert!(p.choose(NOMINAL, FPS, true) < NOMINAL);
    }
}
