//! Bound the optional delay after the ordinary final pass is already ready.
const MAX_EXTRA_NEWTON: u32 = 2;

pub(super) struct Progress {
    authored: f64,
    goal: f64,
    used: u32,
}

impl Progress {
    pub(super) fn new(authored: f32) -> Option<Self> {
        let authored = f64::from(authored);
        if !authored.is_finite() || authored <= 0.0 {
            return None;
        }
        let goal = (2.0 * authored).min(0.5);
        (goal > authored).then_some(Self {
            authored,
            goal,
            used: 0,
        })
    }

    pub(super) fn delay_final(&mut self, advanced: f64, step: u32, cap: u32) -> bool {
        let remaining = MAX_EXTRA_NEWTON - self.used;
        if remaining == 0
            || !advanced.is_finite()
            || advanced < self.authored
            || advanced >= self.goal
        {
            return false;
        }
        // A normal continuation increments step. The final pass uses that same
        // resulting index. Reserve all remaining extra slots and leave the
        // final pass below the original cap, including active-root cases.
        let Some(final_index) = step.checked_add(remaining) else {
            return false;
        };
        if cap != 0 && final_index >= cap {
            return false;
        }
        self.used += 1;
        true
    }

    pub(super) fn used(&self) -> u32 {
        self.used
    }
    pub(super) fn goal(&self) -> f64 {
        self.goal
    }
}

#[cfg(test)]
mod tests {
    use super::Progress;

    #[test]
    fn only_positive_authored_goals_below_half_gain_a_target() {
        for authored in [0.0, -0.1, 0.5, 0.75, f32::NAN, f32::INFINITY] {
            assert!(Progress::new(authored).is_none());
        }
        assert_eq!(Progress::new(0.25).unwrap().goal(), 0.5);
        assert_eq!(Progress::new(0.125).unwrap().goal(), 0.25);
        assert_eq!(Progress::new(0.375).unwrap().goal(), 0.5);
    }

    #[test]
    fn never_delays_before_authored_progress_or_after_target() {
        let mut progress = Progress::new(0.25).unwrap();
        assert!(!progress.delay_final(0.24, 1, 20));
        assert!(!progress.delay_final(f64::NAN, 1, 20));
        assert!(progress.delay_final(0.25, 2, 20));
        assert!(!progress.delay_final(0.5, 3, 20));
        assert_eq!(progress.used(), 1);
    }

    #[test]
    fn only_two_extra_passes_even_without_a_cap() {
        for cap in [0, 8] {
            let mut progress = Progress::new(0.25).unwrap();
            assert!(progress.delay_final(0.25, 5, cap));
            assert!(progress.delay_final(0.30, 6, cap));
            assert!(!progress.delay_final(0.40, 7, cap));
            assert_eq!(progress.used(), 2);
        }
    }

    #[test]
    fn leaves_original_final_routing_when_budget_cannot_reserve_both_passes() {
        let mut progress = Progress::new(0.25).unwrap();
        assert!(!progress.delay_final(0.26, 6, 8));
        assert!(!progress.delay_final(0.26, 7, 8));
        assert!(!progress.delay_final(0.26, u32::MAX - 1, 0));
        assert_eq!(progress.used(), 0);
    }
}
