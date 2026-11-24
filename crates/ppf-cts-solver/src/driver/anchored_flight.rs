use crate::joint_gpu::{BallAnchorGpu, BALL_ANCHOR_FLAG_ACTIVE, WORLD_ENDPOINT};

/// Conservative connectivity from the exact installed FP32 ball records.
/// Auxiliary terms may connect additional bodies, but cannot certify a ball
/// component that lacks a currently prescribed native root.
#[derive(Debug)]
pub(super) struct Components {
    by_body: Vec<usize>,
    count: usize,
}

fn representative(links: &mut [(usize, u8)], mut body: usize) -> usize {
    while links[body].0 != body {
        links[body].0 = links[links[body].0].0;
        body = links[body].0;
    }
    body
}

impl Components {
    pub(super) fn build(bodies: usize, ball: &[BallAnchorGpu]) -> Option<Self> {
        if bodies == 0 || ball.is_empty() {
            return None;
        }
        let mut links = Vec::new();
        links.try_reserve_exact(bodies).ok()?;
        links.extend((0..bodies).map(|body| (body, 0)));
        let mut any_coupling = false;
        for row in ball {
            if row.enabled & BALL_ANCHOR_FLAG_ACTIVE == 0
                || !row.stiffness.is_finite() || row.stiffness <= 0.0
                || row.body_a == WORLD_ENDPOINT || row.body_b == WORLD_ENDPOINT
            {
                continue;
            }
            let (a, b) = (row.body_a as usize, row.body_b as usize);
            if a >= bodies || b >= bodies {
                return None;
            }
            let (mut a, mut b) = (representative(&mut links, a), representative(&mut links, b));
            if a == b { continue; }
            any_coupling = true;
            if links[a].1 < links[b].1 { std::mem::swap(&mut a, &mut b); }
            links[b].0 = a;
            if links[a].1 == links[b].1 { links[a].1 += 1; }
        }
        if !any_coupling { return None; }

        let mut by_body = Vec::new();
        by_body.try_reserve_exact(bodies).ok()?;
        by_body.resize(bodies, usize::MAX);
        let mut count = 0;
        // Produce compact component IDs before replacing each body entry.
        for (body, (parent, _)) in links.iter().copied().enumerate() {
            if body == parent {
                by_body[body] = count;
                count += 1;
            }
        }
        for body in 0..bodies {
            let root = representative(&mut links, body);
            by_body[body] = by_body[root];
        }
        Some(Self { by_body, count })
    }

    pub(super) fn all_prescribed(&self, enabled_roots: &[u32]) -> bool {
        if enabled_roots.len() < self.count
            || enabled_roots.iter().any(|&root| root as usize >= self.by_body.len())
        {
            return false;
        }
        if self.count == 1 {
            return true;
        }
        let mut seen = Vec::new();
        if seen.try_reserve_exact(self.count).is_err() { return false; }
        seen.resize(self.count, false);
        for &root in enabled_roots {
            seen[self.by_body[root as usize]] = true;
        }
        seen.into_iter().all(|covered| covered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(a: u32, b: u32, stiffness: f32, enabled: bool) -> BallAnchorGpu {
        BallAnchorGpu {
            body_a: a,
            body_b: b,
            stiffness,
            enabled: u32::from(enabled),
            record_index: 0,
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
        }
    }

    #[test]
    fn topology_can_be_reused_but_root_membership_is_current() {
        let graph = Components::build(6, &[
            edge(1, 0, 1., true), edge(2, 0, 1., true),
            edge(3, 4, 1., true), edge(5, 4, 1., true), edge(3, 5, 1., true),
        ]).unwrap();
        assert!(graph.all_prescribed(&[2, 3]));
        assert!(graph.all_prescribed(&[4, 0]));
        assert!(!graph.all_prescribed(&[2]));
        assert!(!graph.all_prescribed(&[1, 2]));
        assert!(!graph.all_prescribed(&[2, 7]));
    }

    #[test]
    fn disabled_zero_or_world_rows_cannot_hide_a_free_island() {
        let graph = Components::build(4, &[
            edge(0, 1, 1., true), edge(1, 2, 1., false), edge(1, 3, 0., true),
            edge(WORLD_ENDPOINT, 2, 1., true), edge(WORLD_ENDPOINT, 3, 1., true),
        ]).unwrap();
        assert!(!graph.all_prescribed(&[0, 1, 2]));
        assert!(graph.all_prescribed(&[0, 2, 3]));
        assert!(Components::build(2, &[edge(0, 2, 1., true)]).is_none());
        assert!(Components::build(2, &[edge(0, 1, 0., true)]).is_none());
    }
}
