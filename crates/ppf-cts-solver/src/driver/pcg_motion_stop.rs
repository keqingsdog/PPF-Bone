//! Host-side inertial residual displacement estimate. This is not an animation error bound.
use super::rigid_map::{RigidMap, RIGID_UNSET};

pub(super) struct MotionModel {
    dim: usize,
    cloth: Vec<(usize, Option<f64>)>,
    bodies: Vec<Option<([[f64; 6]; 6], f64)>>,
    free: Vec<bool>,
    body_base: usize,
}

pub(super) struct MotionDecision {
    pub accepted: bool,
    pub motion_groups: Vec<usize>,
    pub max_failed_relative_motion: Option<f64>,
}

impl MotionModel {
    pub fn build(map: &RigidMap, prot: &[f32], masses: &[f32], active: &[bool], free: &[bool], dt: f32) -> Option<Self> {
        if !dt.is_finite() || dt <= 0.0 || prot.len() != map.nrow.checked_mul(3)? || masses.len() != map.nrow || active.len() != map.nrow || free.len() != map.dim || map.vbody.len() != map.nrow || map.cloth_off.len() != map.nrow || map.body_base != map.n_cloth.checked_mul(3)? || map.dim != map.body_base.checked_add(map.n_bodies.checked_mul(6)?)? { return None; }
        let inv_dt2 = 1.0 / (f64::from(dt) * f64::from(dt));
        let mut cloth = Vec::new();
        let mut seen = vec![false; map.n_cloth];
        let mut blocks = vec![[[0.0; 6]; 6]; map.n_bodies];
        let mut radii = vec![0.0f64; map.n_bodies];
        let mut bad = vec![false; map.n_bodies];
        let mut body_active = vec![false; map.n_bodies];
        for i in 0..map.nrow {
            let b = map.vbody[i];
            if b == RIGID_UNSET {
                let o = map.cloth_off[i] as usize;
                if o % 3 != 0 || o.checked_add(3)? > map.body_base || seen[o / 3] { return None; }
                seen[o / 3] = true;
                let movable = free[o..o+3].iter().any(|&x| x);
                let valid = movable && active[i] && masses[i].is_finite() && masses[i] > 0.0;
                cloth.push((o, if valid { Some(1.0 / (f64::from(masses[i]) * inv_dt2)) } else { None }));
            } else {
                let b = b as usize; if b >= map.n_bodies { return None; }
                let p = [f64::from(prot[3*i]), f64::from(prot[3*i+1]), f64::from(prot[3*i+2])];
                if p.iter().any(|x| !x.is_finite()) { bad[b] = true; continue; }
                radii[b] = radii[b].max((p.iter().map(|x| x*x).sum::<f64>()).sqrt());
                if !active[i] { continue; }
                if !masses[i].is_finite() || masses[i] <= 0.0 { bad[b] = true; continue; }
                body_active[b] = true;
                let j = [[1.,0.,0.,0.,p[2],-p[1]], [0.,1.,0.,-p[2],0.,p[0]], [0.,0.,1.,p[1],-p[0],0.]];
                let w = f64::from(masses[i]) * inv_dt2;
                for r in 0..6 { for c in 0..6 { for row in &j { blocks[b][r][c] += w * row[r] * row[c]; } } }
            }
        }
        if seen.iter().any(|&x| !x) { return None; }
        let bodies = (0..map.n_bodies).map(|b| {
            let o = map.body_base + 6*b;
            if bad[b] || !body_active[b] || !free[o..o+6].iter().all(|&v| v) || solve6(blocks[b], [0.;6]).is_none() { None } else { Some((blocks[b], radii[b])) }
        }).collect();
        Some(Self { dim: map.dim, cloth, bodies, free: free.to_vec(), body_base: map.body_base })
    }

    pub fn evaluate(&self, residual: &[f32]) -> Vec<Option<f64>> {
        let mut out = vec![None; 1 + self.bodies.len()];
        if residual.len() != self.dim || residual.iter().any(|x| !x.is_finite()) { return out; }
        let mut cloth_max = 0.0f64; let mut valid = true; let mut any = false;
        for &(o, inv) in &self.cloth {
            if !self.free[o..o+3].iter().any(|&v| v) { continue; }
            if let Some(inv) = inv {
                any = true;
                let e = (0..3).map(|k| if self.free[o+k] { (f64::from(residual[o+k]) * inv).powi(2) } else { 0.0 }).sum::<f64>().sqrt();
                if !e.is_finite() { valid = false; } else { cloth_max = cloth_max.max(e); }
            } else { valid = false; }
        }
        if any && valid { out[0] = Some(cloth_max); }
        for (b, model) in self.bodies.iter().enumerate() {
            if let Some((matrix, radius)) = model {
                let o = self.body_base + 6*b;
                let rhs = std::array::from_fn(|i| f64::from(residual[o+i]));
                if let Some(v) = solve6(*matrix, rhs) {
                    let t = v[..3].iter().map(|x|x*x).sum::<f64>().sqrt();
                    let r = v[3..].iter().map(|x|x*x).sum::<f64>().sqrt();
                    let value = t + radius * r;
                    if value.is_finite() { out[b+1] = Some(value); }
                }
            }
        }
        out
    }
}

// No diagonal jitter: a numerically singular inertia is unavailable.
fn solve6(mut a: [[f64;6];6], mut b: [f64;6]) -> Option<[f64;6]> {
    let scale = a.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    if !scale.is_finite() || scale == 0.0 { return None; }
    for k in 0..6 {
        let p = (k..6).max_by(|&i,&j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if !a[p][k].is_finite() || a[p][k].abs() <= scale * 1e-12 { return None; }
        a.swap(k,p); b.swap(k,p);
        for i in k+1..6 { let f = a[i][k]/a[k][k]; for j in k..6 { a[i][j] -= f*a[k][j]; } b[i] -= f*b[k]; }
    }
    let mut x = [0.;6];
    for i in (0..6).rev() { let tail = (i+1..6).map(|j|a[i][j]*x[j]).sum::<f64>(); x[i] = (b[i]-tail)/a[i][i]; if !x[i].is_finite() { return None; } }
    Some(x)
}

pub(super) fn decide(relative_pass: &[bool], metrics: &[Option<f64>], delta: f64) -> Option<MotionDecision> {
    if relative_pass.len() != metrics.len() || !delta.is_finite() || delta <= 0.0 { return None; }
    let mut accepted = true; let mut motion_groups = Vec::new(); let mut max = None::<f64>;
    for (i, (&relative, metric)) in relative_pass.iter().zip(metrics).enumerate() {
        if relative { continue; }
        match metric {
            Some(x) if x.is_finite() && *x >= 0.0 => { max = Some(max.map_or(*x, |m| m.max(*x))); if *x <= delta { motion_groups.push(i); } else { accepted = false; } }
            _ => accepted = false,
        }
    }
    Some(MotionDecision { accepted, motion_groups, max_failed_relative_motion: max })
}
