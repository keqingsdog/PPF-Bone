//! Host-side mass-weighted tangent fit for an optional PCG initial guess.
use super::rigid_map::{RigidMap, RIGID_UNSET};

pub(super) fn fit(
    map: &RigidMap,
    prot: &[f32],
    eval: &[f32],
    target: &[f32],
    mass: &[f32],
) -> Option<Vec<f32>> {
    let n = map.nrow;
    if prot.len() != 3 * n
        || eval.len() != 3 * n
        || target.len() != 3 * n
        || mass.len() != map.n_bodies
        || map.vbody.len() != n
        || map.cloth_off.len() != n
    {
        return None;
    }
    let mut result = vec![0.0f32; map.dim];
    let mut blocks = vec![[[0.0f64; 7]; 6]; map.n_bodies];
    for i in 0..n {
        let p = [
            prot[3 * i] as f64,
            prot[3 * i + 1] as f64,
            prot[3 * i + 2] as f64,
        ];
        let d = [
            eval[3 * i] as f64 - target[3 * i] as f64,
            eval[3 * i + 1] as f64 - target[3 * i + 1] as f64,
            eval[3 * i + 2] as f64 - target[3 * i + 2] as f64,
        ];
        if p.iter().chain(d.iter()).any(|v| !v.is_finite()) {
            return None;
        }
        let b = map.vbody[i];
        if b == RIGID_UNSET {
            let o = map.cloth_off[i] as usize;
            if o.checked_add(3).map_or(true, |end| end > map.body_base) {
                return None;
            }
            for k in 0..3 {
                result[o + k] = d[k] as f32;
                if !result[o + k].is_finite() {
                    return None;
                }
            }
        } else {
            let b = b as usize;
            if b >= map.n_bodies {
                return None;
            }
            let w = mass[b] as f64;
            if !w.is_finite() || w <= 0.0 {
                return None;
            }
            let j = [
                [1., 0., 0., 0., p[2], -p[1]],
                [0., 1., 0., -p[2], 0., p[0]],
                [0., 0., 1., p[1], -p[0], 0.],
            ];
            for a in 0..6 {
                for c in 0..6 {
                    for k in 0..3 {
                        blocks[b][a][c] += w * j[k][a] * j[k][c];
                    }
                }
                for k in 0..3 {
                    blocks[b][a][6] += w * j[k][a] * d[k];
                }
            }
        }
    }
    for (b, mut a) in blocks.into_iter().enumerate() {
        let scale = (0..6).map(|i| a[i][i].abs()).fold(0.0f64, f64::max);
        if !scale.is_finite() || scale == 0.0 {
            return None;
        }
        for k in 0..6 {
            let pivot = (k..6).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
            if a[pivot][k].abs() <= scale * 1e-12 {
                return None;
            }
            a.swap(k, pivot);
            for i in k + 1..6 {
                let f = a[i][k] / a[k][k];
                for j in k..7 {
                    a[i][j] -= f * a[k][j];
                }
            }
        }
        let mut x = [0.0f64; 6];
        for i in (0..6).rev() {
            let mut v = a[i][6];
            for j in i + 1..6 {
                v -= a[i][j] * x[j];
            }
            x[i] = v / a[i][i];
            if !x[i].is_finite() || !(x[i] as f32).is_finite() {
                return None;
            }
        }
        let o = map.body_base.checked_add(6 * b)?;
        if o + 6 > result.len() {
            return None;
        }
        for i in 0..6 {
            result[o + i] = x[i] as f32;
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn body_map() -> RigidMap {
        RigidMap {
            nrow: 4,
            n_bodies: 1,
            n_cloth: 0,
            dim: 6,
            body_base: 0,
            vbody: vec![0; 4],
            cloth_off: vec![RIGID_UNSET; 4],
            ..Default::default()
        }
    }
    #[test]
    fn recovers_translation_and_rotation() {
        let map = body_map();
        let p = [1., 0., 0., 0., 1., 0., 0., 0., 1., -1., -1., -1.];
        let q = [0.2, -0.3, 0.4, 0.1, -0.2, 0.3];
        let mut d = [0.; 12];
        for i in 0..4 {
            let (x, y, z) = (p[3 * i], p[3 * i + 1], p[3 * i + 2]);
            d[3 * i] = q[0] + z * q[4] - y * q[5];
            d[3 * i + 1] = q[1] - z * q[3] + x * q[5];
            d[3 * i + 2] = q[2] + y * q[3] - x * q[4];
        }
        let fit = fit(&map, &p, &d, &[0.; 12], &[2.]).unwrap();
        for i in 0..6 {
            assert!((fit[i] - q[i]).abs() < 1e-5);
        }
    }
    #[test]
    fn cloth_uses_eval_minus_target() {
        let map = RigidMap {
            nrow: 1,
            n_cloth: 1,
            body_base: 3,
            dim: 3,
            vbody: vec![RIGID_UNSET],
            cloth_off: vec![0],
            ..Default::default()
        };
        assert_eq!(
            fit(&map, &[0.; 3], &[3., 4., 5.], &[1., 1., 1.], &[]).unwrap(),
            vec![2., 3., 4.]
        );
    }
    #[test]
    fn singular_and_invalid_fall_back() {
        let map = body_map();
        assert!(fit(&map, &[0.; 12], &[1.; 12], &[0.; 12], &[1.]).is_none());
        assert!(fit(&map, &[0.; 12], &[f32::NAN; 12], &[0.; 12], &[1.]).is_none());
    }
}
