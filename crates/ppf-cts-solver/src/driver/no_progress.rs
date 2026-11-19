#[derive(Default)]
pub struct RestartTracker { best: Option<f32>, patience: u8 }
impl RestartTracker {
    pub fn clear(&mut self) { self.best = None; self.patience = 0; }
    pub fn observe(&mut self, score: f32) -> Option<f32> {
        if !score.is_finite() { self.clear(); return None; }
        match self.best {
            None => { self.best = Some(score); self.patience = 0; }
            Some(best) if score < best => { self.best = Some(score); self.patience = 0; }
            Some(best) => { self.patience += 1; if self.patience >= 16 { return Some(best); } }
        }
        None
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn budget_and_improvement() {
        let mut t=RestartTracker::default(); assert_eq!(t.observe(2.),None);
        for _ in 0..15 { assert_eq!(t.observe(2.),None); }
        assert_eq!(t.observe(2.),Some(2.));
        assert_eq!(t.observe(1.),None);
        for _ in 0..15 { assert_eq!(t.observe(2.),None); }
        assert_eq!(t.observe(2.),Some(1.));
    }
    #[test] fn clear_and_nonfinite_reset() {
        let mut t=RestartTracker::default(); t.observe(2.); for _ in 0..15 { t.observe(2.); }
        t.clear(); assert_eq!(t.observe(2.),None); assert_eq!(t.observe(f32::NAN),None);
        assert_eq!(t.observe(2.),None); for _ in 0..15 { assert_eq!(t.observe(2.),None); }
        assert_eq!(t.observe(2.),Some(2.));
    }
}
