use std::collections::{BTreeMap, BTreeSet};

#[path = "compact_full_body_factor.rs"]
mod compact;
pub use self::compact::Cache as SymbolicCache;

pub type Block = [[f64; 6]; 6];
pub type BodyVector = [f64; 6];
type Key = (usize, usize);

#[derive(Clone, Copy)]
pub struct Budget {
    pub bytes: usize,
    pub fill_blocks: usize,
}

#[derive(Debug)]
pub struct Stats {
    pub original_edges: usize,
    pub fill_edges: usize,
    pub max_front: usize,
    pub charged_bytes: usize,
    pub max_asymmetry: f64,
}

pub type Result<T> = std::result::Result<T, &'static str>;

pub struct Factor {
    storage: Storage,
    pub stats: Stats,
}

enum Storage {
    Legacy(Vec<Column>),
    Compact(compact::Compact),
}

struct Column {
    body: usize,
    diagonal: Block,
    multipliers: Vec<(usize, Block)>,
}

fn reserve<T>(length: usize) -> Result<Vec<T>> {
    let mut items = Vec::new();
    items.try_reserve_exact(length).map_err(|_| "workspace allocation")?;
    Ok(items)
}

fn charge(bodies: usize, edges: usize, budget: Budget) -> Result<usize> {
    // Includes overlapping maps, adjacency, retained factors and the largest
    // front workspace. Charge total fill even after its live edge is retired.
    let bytes = bodies.checked_mul(4096)
        .and_then(|n| edges.checked_mul(1536).and_then(|m| n.checked_add(m)))
        .ok_or("workspace extent")?;
    if bytes > budget.bytes { return Err("workspace budget"); }
    Ok(bytes)
}

fn transpose(a: Block) -> Block {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

fn cholesky(a: Block) -> Result<Block> {
    let mut l = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut value = 0.5 * (a[i][j] + a[j][i]);
            for k in 0..j {
                value -= l[i][k] * l[j][k];
            }
            if !value.is_finite() || (i == j && value <= 0.0) {
                return Err("nonpositive or nonfinite pivot");
            }
            l[i][j] = if i == j { value.sqrt() } else { value / l[j][j] };
            if !l[i][j].is_finite() { return Err("nonfinite triangular factor"); }
        }
    }
    Ok(l)
}

fn solve_block(l: &Block, mut x: BodyVector) -> BodyVector {
    for i in 0..6 {
        for j in 0..i { x[i] -= l[i][j] * x[j]; }
        x[i] /= l[i][i];
    }
    for i in (0..6).rev() {
        for j in i + 1..6 { x[i] -= l[j][i] * x[j]; }
        x[i] /= l[i][i];
    }
    x
}

impl Factor {
    /// Free-body blocks in projected row/column order. Missing transpose
    /// blocks are zero by definition, and their skew is included in stats.
    pub fn build(
        bodies: usize, directed: &BTreeMap<Key, [[f32; 6]; 6]>, budget: Budget,
    ) -> Result<Self> {
        Self::build_compute_bounded(bodies, directed, budget, usize::MAX)
    }

    pub fn build_compute_bounded(
        bodies: usize, directed: &BTreeMap<Key, [[f32; 6]; 6]>, budget: Budget,
        mut remaining_updates: usize,
    ) -> Result<Self> {
        if bodies == 0 { return Err("empty domain"); }
        charge(bodies, directed.len(), budget)?;
        let zero = [[0.0; 6]; 6];
        let mut entries = BTreeMap::<Key, Block>::new();
        let mut stats = Stats {
            original_edges: 0, fill_edges: 0, max_front: 0, charged_bytes: 0,
            max_asymmetry: 0.0,
        };
        for (&(row, col), block) in directed {
            if row >= bodies || col >= bodies || block.iter().flatten().any(|x| !x.is_finite()) {
                return Err("invalid directed block");
            }
            let key = (row.min(col), row.max(col));
            if entries.contains_key(&key) { continue; }
            let a = directed.get(&key).unwrap_or(&zero);
            let b = directed.get(&(key.1, key.0)).unwrap_or(&zero);
            let mut reduced = [[0.0; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    let (x, y) = (f64::from(a[i][j]), f64::from(b[j][i]));
                    stats.max_asymmetry = stats.max_asymmetry.max((x-y).abs());
                    reduced[i][j] = 0.5 * (x+y);
                }
            }
            if key.0 == key.1 || reduced.iter().flatten().any(|&x| x != 0.0) {
                entries.insert(key, reduced);
            }
        }
        let mut adjacency = reserve(bodies)?;
        adjacency.resize_with(bodies, BTreeSet::new);
        for &(a, b) in entries.keys() {
            if a != b {
                adjacency[a].insert(b);
                adjacency[b].insert(a);
                stats.original_edges += 1;
            }
        }
        stats.charged_bytes = charge(bodies, directed.len(), budget)?;
        let mut queue: BTreeSet<_> = adjacency.iter().enumerate()
            .map(|(body, neighbors)| (neighbors.len(), body)).collect();
        let mut columns = reserve(bodies)?;
        while let Some((degree, pivot)) = queue.pop_first() {
            let updates = degree.checked_add(1).and_then(|next| degree.checked_mul(next))
                .ok_or("work extent")? / 2;
            remaining_updates = remaining_updates.checked_sub(updates).ok_or("work budget")?;
            stats.max_front = stats.max_front.max(degree);
            let l = cholesky(entries.remove(&(pivot, pivot)).ok_or("missing diagonal")?)?;
            let neighbors = std::mem::take(&mut adjacency[pivot]);
            let mut front = reserve(degree)?;
            let mut multipliers = reserve(degree)?;
            for body in neighbors {
                queue.remove(&(adjacency[body].len(), body));
                adjacency[body].remove(&pivot);
                let original = if pivot < body {
                    transpose(entries.remove(&(pivot, body)).ok_or("missing edge")?)
                } else {
                    entries.remove(&(body, pivot)).ok_or("missing edge")?
                };
                let w = original.map(|row| solve_block(&l, row));
                if w.iter().flatten().any(|x| !x.is_finite()) { return Err("nonfinite multiplier"); }
                front.push(original);
                multipliers.push((body, w));
            }
            for (idx, (a, w)) in multipliers.iter().enumerate() {
                for (offset, (b, _)) in multipliers[idx..].iter().enumerate() {
                    let key = (*a, *b);
                    if !entries.contains_key(&key) {
                        if a == b { return Err("missing diagonal"); }
                        stats.fill_edges = stats.fill_edges.checked_add(1).ok_or("fill extent")?;
                        if stats.fill_edges > budget.fill_blocks { return Err("fill budget"); }
                        let total = directed.len().checked_add(stats.fill_edges).ok_or("fill extent")?;
                        stats.charged_bytes = charge(bodies, total, budget)?;
                        entries.insert(key, zero.map(|row| row.map(f64::from)));
                        adjacency[*a].insert(*b);
                        adjacency[*b].insert(*a);
                    }
                    let raw = &front[idx + offset];
                    let current = entries.get_mut(&key).expect("the live front contains its blocks");
                    for i in 0..6 {
                        for j in 0..6 {
                            let correction: f64 = (0..6).map(|k| w[i][k] * raw[j][k]).sum();
                            current[i][j] -= correction;
                        }
                    }
                }
            }
            for (body, _) in &multipliers {
                queue.insert((adjacency[*body].len(), *body));
            }
            columns.push(Column { body: pivot, diagonal: l, multipliers });
        }
        Ok(Self { storage: Storage::Legacy(columns), stats })
    }

    pub fn build_compact(
        bodies: usize,
        directed: &BTreeMap<Key, [[f32; 6]; 6]>,
        budget: Budget,
        work: usize,
        cache: &mut SymbolicCache,
    ) -> Result<(Self, bool)> {
        let (factor, stats, hit) = compact::Compact::build(bodies, directed, budget, work, cache)?;
        Ok((Self { storage: Storage::Compact(factor), stats }, hit))
    }

    pub fn solve(&self, rhs: &[BodyVector]) -> Result<Vec<BodyVector>> {
        let columns = match &self.storage {
            Storage::Legacy(columns) => columns,
            Storage::Compact(factor) => return factor.solve(rhs),
        };
        if rhs.len() != columns.len() || rhs.iter().flatten().any(|x| !x.is_finite()) {
            return Err("invalid RHS");
        }
        let mut x = reserve(rhs.len())?;
        x.extend_from_slice(rhs);
        for c in columns {
            let y = x[c.body];
            for (other, w) in &c.multipliers {
                for i in 0..6 {
                    x[*other][i] -= (0..6).map(|j| w[i][j] * y[j]).sum::<f64>();
                }
            }
        }
        for c in columns {
            x[c.body] = solve_block(&c.diagonal, x[c.body]);
        }
        for c in columns.iter().rev() {
            for (other, w) in &c.multipliers {
                let z = x[*other];
                for i in 0..6 {
                    x[c.body][i] -= (0..6).map(|j| w[j][i] * z[j]).sum::<f64>();
                }
            }
        }
        if x.iter().flatten().any(|value| !value.is_finite()) { return Err("nonfinite solve"); }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal(scale: f32) -> [[f32; 6]; 6] {
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { scale } else { 0.0 }))
    }

    #[test]
    fn cyclic_graph_retains_every_edge_and_solves() {
        let bodies = 192;
        let mut matrix = BTreeMap::new();
        for a in 0..bodies {
            matrix.insert((a, a), diagonal(7.0));
            for b in [(a + 1) % bodies, (a + 16) % bodies] {
                let mut link = diagonal(-1.0);
                link[0][4] = 0.10;
                link[5][1] = -0.20;
                let reciprocal = std::array::from_fn(|i| std::array::from_fn(|j| link[j][i]));
                matrix.insert((a, b), link);
                matrix.insert((b, a), reciprocal);
            }
        }
        let budget = Budget { bytes: 64 << 20, fill_blocks: 8192 };
        let started = std::time::Instant::now();
        let factor = Factor::build(bodies, &matrix, budget).unwrap();
        let factor_elapsed = started.elapsed();
        assert_eq!(factor.stats.original_edges, 2 * bodies);
        assert!(factor.stats.fill_edges > 0);
        let expected: Vec<BodyVector> = (0..bodies)
            .map(|a| std::array::from_fn(|i| ((a*7+i*11)%31) as f64 - 15.0)).collect();
        let mut rhs = vec![[0.0; 6]; bodies];
        for (&(a,b), m) in &matrix {
            for i in 0..6 {
                rhs[a][i] += (0..6).map(|j| f64::from(m[i][j]) * expected[b][j]).sum::<f64>();
            }
        }
        let started = std::time::Instant::now();
        let solution = factor.solve(&rhs).unwrap();
        eprintln!("synthetic cyclic chain factor={factor_elapsed:?} solve={:?} stats={:?}",
            started.elapsed(), factor.stats);
        assert!(solution.iter().flatten().zip(expected.iter().flatten()).all(|(a,b)| (a-b).abs() < 1e-11));
        assert!(matches!(Factor::build(bodies, &matrix, Budget { fill_blocks: 0, ..budget }), Err("fill budget")));
    }

    #[test]
    fn refuses_bad_pivot_and_tiny_budget() {
        let matrix = BTreeMap::from([((0, 0), diagonal(-1.0))]);
        assert!(matches!(Factor::build(1, &matrix, Budget { bytes: 128, fill_blocks: 0 }), Err("workspace budget")));
        assert!(matches!(Factor::build(1, &matrix, Budget { bytes: 1 << 20, fill_blocks: 0 }),
            Err("nonpositive or nonfinite pivot")));
    }
}

#[cfg(test)]
mod native_budget_tests {
    use super::*;

    #[test]
    fn existing_edges_spend_work_even_if_no_new_fill_is_required() {
        let diagonal = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 4.0 } else { 0.0 }));
        let edge = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { -1.0 } else { 0.0 }));
        let graph = [((0, 0), diagonal), ((1, 1), diagonal),
            ((0, 1), edge), ((1, 0), edge)].into_iter().collect();
        let budget = Budget { bytes: 1 << 20, fill_blocks: 0 };
        assert_eq!(Factor::build_compute_bounded(2, &graph, budget, 0).err(), Some("work budget"));
        assert!(Factor::build_compute_bounded(2, &graph, budget, 1).is_ok());
        let budget = Budget { bytes: 1, fill_blocks: 0 };
        assert_eq!(Factor::build_compute_bounded(2, &graph, budget, 1).err(), Some("workspace budget"));
    }
}
