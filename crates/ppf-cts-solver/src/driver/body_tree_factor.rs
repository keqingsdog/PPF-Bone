//! Independent f64 block-tree factor. No production/backend integration.
use std::collections::{BTreeMap, VecDeque};
pub type Block = [[f64; 6]; 6];
#[derive(Debug, Clone)]
pub struct Pair {
    pub a: usize,
    pub b: usize,
    pub ab: Block,
}
#[derive(Debug, Clone)]
pub struct Unsupported(pub String);
type R<T> = Result<T, Unsupported>;
fn bad(s: &str) -> Unsupported {
    Unsupported(s.into())
}
fn transpose(a: &Block) -> Block {
    let mut t = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            t[i][j] = a[j][i];
        }
    }
    t
}
fn finite(a: &Block) -> bool {
    a.iter().flatten().all(|x| x.is_finite())
}
fn chol(a: &Block) -> R<Block> {
    let mut l = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut v = a[i][j];
            for k in 0..j {
                v -= l[i][k] * l[j][k];
            }
            if i == j {
                if !v.is_finite() || v <= 0. {
                    return Err(bad("nonpositive/nonfinite Schur pivot"));
                }
                l[i][j] = v.sqrt();
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    if !finite(&l) {
        return Err(bad("nonfinite factor"));
    }
    Ok(l)
}
fn solve(l: &Block, b: &[f64; 6]) -> [f64; 6] {
    let mut x = *b;
    for i in 0..6 {
        for j in 0..i {
            x[i] -= l[i][j] * x[j];
        }
        x[i] /= l[i][i];
    }
    for i in (0..6).rev() {
        for j in i + 1..6 {
            x[i] -= l[j][i] * x[j];
        }
        x[i] /= l[i][i];
    }
    x
}
#[derive(Debug)]
pub struct Tree {
    pub roots: Vec<usize>,
    parent: Vec<Option<usize>>,
    order: Vec<usize>,
    l: Vec<Block>,
    w: Vec<Block>,
}
pub struct TreeView<'a> { pub parent: &'a [Option<usize>], pub order: &'a [usize], pub l: &'a [Block], pub w: &'a [Block] }
impl Tree {
    pub fn view(&self) -> TreeView<'_> { TreeView { parent: &self.parent, order: &self.order, l: &self.l, w: &self.w } }
    /// Diagonals are full symmetric blocks. Pair.ab is A[a,b], A[b,a]=ab^T.
    /// Every supplied pair is structural, including exact-zero blocks. Duplicate
    /// unordered pairs are summed after orienting them; none are threshold-dropped.
    pub fn factor(diag: &[Block], pairs: &[Pair]) -> R<Self> {
        let n = diag.len();
        if n == 0 {
            return Err(bad("empty graph"));
        }
        for d in diag {
            if !finite(d) {
                return Err(bad("nonfinite diagonal"));
            }
            for i in 0..6 {
                for j in 0..6 {
                    if d[i][j] != d[j][i] {
                        return Err(bad("asymmetric diagonal"));
                    }
                }
            }
        }
        let mut edge = BTreeMap::<(usize, usize), Block>::new();
        for p in pairs {
            if p.a >= n || p.b >= n || p.a == p.b || !finite(&p.ab) {
                return Err(bad("invalid pair"));
            }
            let (a, b, m) = if p.a < p.b {
                (p.a, p.b, p.ab)
            } else {
                (p.b, p.a, transpose(&p.ab))
            };
            let d = edge.entry((a, b)).or_insert([[0.; 6]; 6]);
            for i in 0..6 {
                for j in 0..6 {
                    d[i][j] += m[i][j];
                }
            }
            if !finite(d) {
                return Err(bad("pair accumulation overflow"));
            }
        }
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edge.keys() {
            adj[a].push(b);
            adj[b].push(a);
        }
        let mut seen = vec![false; n];
        let mut roots = Vec::new();
        let mut parent = vec![None; n];
        let mut order = Vec::new();
        let mut degree: Vec<usize> = adj.iter().map(Vec::len).collect();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            let mut comp = vec![start];
            seen[start] = true;
            let mut at = 0;
            while at < comp.len() {
                let u = comp[at];
                for &v in &adj[u] {
                    if !seen[v] {
                        seen[v] = true;
                        comp.push(v);
                    }
                }
                at += 1;
            }
            let ec: usize = comp.iter().map(|&u| adj[u].len()).sum::<usize>() / 2;
            if ec + 1 != comp.len() {
                return Err(bad("cycle: forest required"));
            }
            // Repeatedly peel leaves; the remaining one or two vertices are
            // exactly the minimum-eccentricity centers. Preserve the ID tie break.
            let mut leaves: VecDeque<usize> = comp.iter().copied().filter(|&u| degree[u] <= 1).collect();
            let mut remaining = comp.len();
            while remaining > 2 {
                let count = leaves.len();
                if count == 0 { return Err(bad("cycle: forest required")); }
                remaining -= count;
                for _ in 0..count {
                    let u = leaves.pop_front().unwrap();
                    degree[u] = 0;
                    for &v in &adj[u] {
                        if degree[v] > 0 {
                            degree[v] -= 1;
                            if degree[v] == 1 { leaves.push_back(v); }
                        }
                    }
                }
            }
            let root = *leaves.iter().min().ok_or_else(|| bad("missing tree center"))?;
            roots.push(root);
            let mut q = VecDeque::from([root]);
            while let Some(u) = q.pop_front() {
                order.push(u);
                for &v in &adj[u] {
                    if Some(v) != parent[u] {
                        parent[v] = Some(u);
                        q.push_back(v);
                    }
                }
            }
        }
        let mut d = diag.to_vec();
        let mut l = vec![[[0.; 6]; 6]; n];
        let mut w = l.clone();
        for &u in order.iter().rev() {
            l[u] = chol(&d[u])?;
            if let Some(p) = parent[u] {
                let block = if u < p {
                    edge[&(u, p)]
                } else {
                    transpose(&edge[&(p, u)])
                };
                for col in 0..6 {
                    let mut rhs = [0.; 6];
                    for r in 0..6 {
                        rhs[r] = block[r][col];
                    }
                    let x = solve(&l[u], &rhs);
                    for r in 0..6 {
                        w[u][r][col] = x[r];
                    }
                }
                // Symmetric Schur update computes lower triangle once and mirrors it;
                // this is symmetric factor arithmetic, not deletion of input asymmetry.
                for i in 0..6 {
                    for j in 0..=i {
                        let mut v = 0.;
                        for r in 0..6 {
                            v += block[r][i] * w[u][r][j];
                        }
                        let x = d[p][i][j] - v;
                        d[p][i][j] = x;
                        d[p][j][i] = x;
                    }
                }
                if !finite(&w[u]) || !finite(&d[p]) {
                    return Err(bad("nonfinite elimination"));
                }
            }
        }
        Ok(Self {
            roots,
            parent,
            order,
            l,
            w,
        })
    }
    pub fn apply(&self, rhs: &[f64]) -> R<Vec<f64>> {
        let n = self.l.len();
        if rhs.len() != 6 * n || !rhs.iter().all(|x| x.is_finite()) {
            return Err(bad("invalid rhs"));
        }
        let mut q = vec![[0.; 6]; n];
        for i in 0..n {
            q[i].copy_from_slice(&rhs[6 * i..6 * i + 6]);
        }
        for &u in self.order.iter().rev() {
            if let Some(p) = self.parent[u] {
                for j in 0..6 {
                    let mut t = 0.;
                    for i in 0..6 {
                        t += self.w[u][i][j] * q[u][i];
                    }
                    q[p][j] -= t;
                }
            }
        }
        let mut x = vec![[0.; 6]; n];
        for &u in &self.order {
            x[u] = solve(&self.l[u], &q[u]);
            if let Some(p) = self.parent[u] {
                for i in 0..6 {
                    for j in 0..6 {
                        x[u][i] -= self.w[u][i][j] * x[p][j];
                    }
                }
            }
        }
        let flat: Vec<f64> = x.into_iter().flatten().collect();
        if !flat.iter().all(|x| x.is_finite()) {
            return Err(bad("nonfinite apply"));
        }
        Ok(flat)
    }
}
/// Diagnostic cast ONLY; no solver acceptance claim. Overflow explicitly fails.
pub fn diagnostic_cast32(x: &[f64]) -> R<Vec<f32>> {
    let y: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    if !y.iter().all(|v| v.is_finite()) {
        return Err(bad("cast overflow/nonfinite"));
    }
    Ok(y)
}
