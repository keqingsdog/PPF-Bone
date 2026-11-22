//! Conservative continuous guards for a pin-authorized face/edge crossing.
//! Point order is hard0, hard1, free, pull0, pull1.
type Point = [f64; 3];

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn det(a: Point, b: Point, c: Point) -> (f64, f64) {
    let terms = [
        a[0] * b[1] * c[2], a[1] * b[2] * c[0], a[2] * b[0] * c[1],
        -a[2] * b[1] * c[0], -a[1] * b[0] * c[2], -a[0] * b[2] * c[1],
    ];
    let value = terms.iter().sum();
    let magnitude: f64 = terms.iter().map(|x| x.abs()).sum();
    (value, magnitude)
}

// The cubic Bernstein coefficients of orient(a,b,c,d), multiplied by 3
// for the two interior coefficients. Every coefficient must have a safely
// separated sign for a proof that the orientation stays nonzero.
fn orientation_controls(old: &[Point; 5], new: &[Point; 5], ids: [usize; 4]) -> [(f64, f64); 4] {
    let vectors = |p: &[Point; 5]| {
        [sub(p[ids[1]], p[ids[0]]), sub(p[ids[2]], p[ids[0]]), sub(p[ids[3]], p[ids[0]])]
    };
    let a = vectors(old);
    let b = vectors(new);
    let mix = |i: usize, j: usize, k: usize| {
        det(if i == 0 { a[0] } else { b[0] },
            if j == 0 { a[1] } else { b[1] },
            if k == 0 { a[2] } else { b[2] })
    };
    let sum = |parts: [(f64, f64); 3]| {
        (parts.iter().map(|x| x.0).sum(), parts.iter().map(|x| x.1).sum())
    };
    [
        mix(0, 0, 0),
        sum([mix(1, 0, 0), mix(0, 1, 0), mix(0, 0, 1)]),
        sum([mix(1, 1, 0), mix(1, 0, 1), mix(0, 1, 1)]),
        mix(1, 1, 1),
    ]
}

fn sign(value: (f64, f64)) -> i8 {
    let (v, bound) = value;
    // Include roundoff from coordinate subtraction and determinant arithmetic.
    let error = bound * f64::EPSILON * 128.0;
    if !v.is_finite() || !bound.is_finite() || v.abs() <= error {
        0
    } else if v > 0.0 {
        1
    } else {
        -1
    }
}

fn stable(c: &[(f64, f64); 4]) -> bool {
    let s = sign(c[0]);
    s != 0 && c.iter().all(|&x| sign(x) == s)
}

pub(crate) fn authorized_transition(
    accepted: [[f32; 3]; 5],
    candidate: [[f32; 3]; 5],
    prior_authorized: bool,
) -> bool {
    if accepted.iter().chain(candidate.iter()).flatten().any(|x| !x.is_finite()) {
        return false;
    }
    let coordinate_max = accepted.iter().chain(candidate.iter()).flatten().fold(0.0_f64, |m, x| m.max(f64::from(*x).abs()));
    let old = accepted.map(|p| p.map(f64::from));
    let new = candidate.map(|p| p.map(f64::from));
    let roundoff_floor = coordinate_max.powi(3) * f64::EPSILON * 4096.0;
    let controls = |ids| orientation_controls(&old, &new, ids).map(|(value, magnitude)| (value, magnitude + roundoff_floor / (f64::EPSILON * 128.0)));
    let face_0 = controls([0, 1, 2, 3]);
    let face_1 = controls([0, 1, 2, 4]);
    let hard_edge = controls([3, 4, 0, 1]);
    let free_edge_1 = controls([3, 4, 1, 2]);
    let free_edge_0 = controls([3, 4, 2, 0]);
    if !stable(&face_0) || !stable(&face_1)
        || !stable(&free_edge_1) || !stable(&free_edge_0)
    {
        return false;
    }

    let hard_new = sign(hard_edge[3]);
    let inside = hard_new != 0
        && hard_new == sign(free_edge_1[3])
        && hard_new == sign(free_edge_0[3])
        && sign(face_0[3]) == -sign(face_1[3]);
    if !inside {
        return false;
    }

    if prior_authorized {
        // Historical ownership is supplied only for a previously accepted
        // matching pair. A fixed sign proves it did not leave and re-enter.
        stable(&hard_edge)
    } else {
        // One Bernstein sign change proves a single crossing of the
        // authorized edge. The other support-plane signs remain fixed.
        let signs = hard_edge.map(sign);
        signs[0] != 0
            && signs[3] == -signs[0]
            && signs.iter().all(|&s| s != 0)
            && signs.windows(2).filter(|pair| pair[0] != pair[1]).count() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::authorized_transition;

    fn scene(y: f32) -> [[f32; 3]; 5] {
        [[-1., 0., 0.], [1., 0., 0.], [0., 2., 0.],
         [0., y, -1.], [0., y, 1.]]
    }

    #[test]
    fn enters_via_hard_edge_and_continues() {
        assert!(authorized_transition(scene(-0.8), scene(0.5), false));
        assert!(authorized_transition(scene(0.5), scene(0.7), true));
        assert!(!authorized_transition(scene(0.5), scene(0.7), false));
    }

    #[test]
    fn rejects_free_boundary_endpoint_crossing_and_degeneracy() {
        assert!(!authorized_transition(scene(3.), scene(0.5), false));
        let mut same_side = scene(-0.8);
        same_side[3][2] = 1.0;
        assert!(!authorized_transition(same_side, scene(0.5), false));

        // The edge pierces both endpoint poses. At t=0.5 its intersection
        // touches the moving hard1-to-free boundary and returns inside.
        let start = [[-1., 0., 0.], [1., 0., 0.], [1., 1., 0.],
                     [0.75, 0.125, -1.], [0.75, 0.125, 1.]];
        let end = [[-1., 0., 0.], [1., 0., 0.], [5., 1., 0.],
                   [2.25, 0.375, -1.], [2.25, 0.375, 1.]];
        assert!(!authorized_transition(start, end, true));
        let mut end = scene(0.5);
        end[3][2] = 1.;
        assert!(!authorized_transition(scene(-1.), end, false));
        let mut degenerate = scene(0.5);
        degenerate[2] = [0., 0., 0.];
        assert!(!authorized_transition(scene(-1.), degenerate, false));
        degenerate[2][0] = f32::NAN;
        assert!(!authorized_transition(scene(-1.), degenerate, false));
    }
}

