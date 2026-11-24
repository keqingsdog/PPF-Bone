//! Minimum-degree symbolic plan and contiguous block Cholesky storage.

use super::{charge, cholesky, reserve, transpose, Block, BodyVector, Budget, Key, Result, Stats};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;

const ZERO: Block = [[0.0; 6]; 6];

#[derive(Clone, Copy)]
struct Edge {
    body: usize,
    slot: u32,
}

struct Column {
    body: usize,
    diagonal: u32,
    end: usize,
}

struct Plan {
    keys: Vec<Key>,
    columns: Vec<Column>,
    front: Vec<Edge>,
    updates: Vec<u32>,
    blocks: usize,
    original_edges: usize,
    fill_edges: usize,
    max_front: usize,
}

/// Retains only an exact structural plan. Every factor owns its current values.
#[derive(Default)]
pub struct Cache {
    plan: Option<Arc<Plan>>,
}

pub(super) struct Compact {
    plan: Arc<Plan>,
    values: Vec<Block>,
}

fn push<T>(out: &mut Vec<T>, value: T) -> Result<()> {
    out.try_reserve(1).map_err(|_| "workspace allocation")?;
    out.push(value);
    Ok(())
}

fn slots(
    bodies: usize,
    directed: &BTreeMap<Key, [[f32; 6]; 6]>,
) -> Result<(Vec<(Key, Block)>, f64)> {
    let mut source = reserve(directed.len())?;
    let mut skew = 0.0_f64;
    let zero = [[0.0_f32; 6]; 6];
    for (&(a, b), block) in directed {
        if a >= bodies || b >= bodies || block.iter().flatten().any(|value| !value.is_finite()) {
            return Err("invalid directed block");
        }
        if a > b && directed.contains_key(&(b, a)) {
            continue;
        }
        let key = (a.min(b), a.max(b));
        let upper = if a <= b { block } else { &zero };
        let lower = if a >= b { block } else {
            directed.get(&(b, a)).unwrap_or(&zero)
        };
        let symmetric = std::array::from_fn(|i| std::array::from_fn(|j| {
            let x = f64::from(upper[i][j]);
            let y = f64::from(lower[j][i]);
            skew = skew.max((x - y).abs());
            0.5 * (x + y)
        }));
        if a == b || symmetric.iter().flatten().any(|&value| value != 0.0) {
            source.push((key, symmetric));
        }
    }
    source.sort_unstable_by_key(|&(key, _)| key);
    Ok((source, skew))
}

fn add_edge(row: &mut Vec<Edge>, position: usize, edge: Edge) -> Result<()> {
    row.try_reserve(1).map_err(|_| "workspace allocation")?;
    row.insert(position, edge);
    Ok(())
}

impl Plan {
    fn matches(&self, source: &[(Key, Block)], bodies: usize, budget: Budget, work: usize) -> bool {
        self.columns.len() == bodies
            && self.fill_edges <= budget.fill_blocks
            && self.updates.len() <= work
            && self.keys.len() == source.len()
            && self.keys.iter().zip(source).all(|(key, (other, _))| key == other)
    }

    fn build(bodies: usize, source: &[(Key, Block)], directed: usize,
             budget: Budget, work: usize) -> Result<Self> {
        let mut adjacency = reserve(bodies)?;
        adjacency.resize_with(bodies, Vec::<Edge>::new);
        let mut diagonal = reserve(bodies)?;
        diagonal.resize(bodies, u32::MAX);
        let mut keys = reserve(source.len())?;
        for (slot, &((a, b), _)) in source.iter().enumerate() {
            let slot = u32::try_from(slot).map_err(|_| "workspace extent")?;
            keys.push((a, b));
            if a == b {
                diagonal[a] = slot;
            } else {
                push(&mut adjacency[a], Edge { body: b, slot })?;
                push(&mut adjacency[b], Edge { body: a, slot })?;
            }
        }
        if diagonal.contains(&u32::MAX) {
            return Err("missing diagonal");
        }
        for row in &mut adjacency {
            row.sort_unstable_by_key(|edge| edge.body);
        }
        let mut queue = BinaryHeap::new();
        queue.try_reserve(bodies).map_err(|_| "workspace allocation")?;
        queue.extend(adjacency.iter().enumerate().map(|(body, row)| Reverse((row.len(), body))));
        let mut columns = reserve(bodies)?;
        let mut front = reserve(source.len().saturating_sub(bodies))?;
        let mut updates = Vec::new();
        let mut blocks = source.len();
        let mut max_front = 0usize;
        while let Some(Reverse((degree, pivot))) = queue.pop() {
            if diagonal[pivot] == u32::MAX || degree != adjacency[pivot].len() {
                continue;
            }
            let pivot_diagonal = std::mem::replace(&mut diagonal[pivot], u32::MAX);
            let added_work = degree.checked_add(1).and_then(|n| degree.checked_mul(n))
                .ok_or("work extent")? / 2;
            if added_work > work.saturating_sub(updates.len()) {
                return Err("work budget");
            }
            updates.try_reserve(added_work).map_err(|_| "workspace allocation")?;
            front.try_reserve(degree).map_err(|_| "workspace allocation")?;
            queue.try_reserve(degree).map_err(|_| "workspace allocation")?;
            max_front = max_front.max(degree);
            let neighbors = std::mem::take(&mut adjacency[pivot]);
            for &Edge { body, .. } in &neighbors {
                let row = &mut adjacency[body];
                let index = row.binary_search_by_key(&pivot, |edge| edge.body)
                    .map_err(|_| "missing edge")?;
                row.remove(index);
            }
            for (index, &Edge { body: a, .. }) in neighbors.iter().enumerate() {
                updates.push(diagonal[a]);
                for &Edge { body: b, .. } in &neighbors[index + 1..] {
                    let target = match adjacency[a].binary_search_by_key(&b, |edge| edge.body) {
                        Ok(i) => adjacency[a][i].slot,
                        Err(i) => {
                            let fill = blocks - source.len() + 1;
                            if fill > budget.fill_blocks {
                                return Err("fill budget");
                            }
                            charge(bodies, directed.checked_add(fill).ok_or("fill extent")?, budget)?;
                            let slot = u32::try_from(blocks).map_err(|_| "workspace extent")?;
                            blocks += 1;
                            let other = adjacency[b].binary_search_by_key(&a, |edge| edge.body)
                                .err().ok_or("inconsistent edge")?;
                            add_edge(&mut adjacency[a], i, Edge { body: b, slot })?;
                            add_edge(&mut adjacency[b], other, Edge { body: a, slot })?;
                            slot
                        }
                    };
                    updates.push(target);
                }
            }
            for &Edge { body, .. } in &neighbors {
                queue.push(Reverse((adjacency[body].len(), body)));
            }
            front.extend(neighbors);
            columns.push(Column { body: pivot, diagonal: pivot_diagonal, end: front.len() });
        }
        if columns.len() != bodies {
            return Err("incomplete ordering");
        }
        Ok(Self { keys, columns, front, updates, blocks, original_edges: source.len() - bodies,
                  fill_edges: blocks - source.len(), max_front })
    }
}

#[inline]
fn forward(l: &Block, mut x: BodyVector) -> BodyVector {
    for i in 0..6 {
        for k in 0..i {
            x[i] -= l[i][k] * x[k];
        }
        x[i] /= l[i][i];
    }
    x
}

#[inline]
fn backward(l: &Block, mut x: BodyVector) -> BodyVector {
    for i in (0..6).rev() {
        for k in i + 1..6 {
            x[i] -= l[k][i] * x[k];
        }
        x[i] /= l[i][i];
    }
    x
}

#[inline]
fn dot(a: &BodyVector, b: &BodyVector) -> f64 {
    (0..6).map(|k| a[k] * b[k]).sum()
}

impl Compact {
    pub(super) fn build(
        bodies: usize, directed: &BTreeMap<Key, [[f32; 6]; 6]>, budget: Budget,
        work: usize, cache: &mut Cache,
    ) -> Result<(Self, Stats, bool)> {
        if bodies == 0 {
            return Err("empty domain");
        }
        charge(bodies, directed.len(), budget)?;
        let (source, max_asymmetry) = slots(bodies, directed)?;
        let cached = cache.plan.as_ref().filter(|plan| plan.matches(&source, bodies, budget, work));
        let hit = cached.is_some();
        let plan = if let Some(plan) = cached {
            Arc::clone(plan)
        } else {
            cache.plan = None;
            Arc::new(Plan::build(bodies, &source, directed.len(), budget, work)?)
        };
        let charged_bytes = charge(bodies, directed.len().checked_add(plan.fill_edges)
            .ok_or("fill extent")?, budget)?;
        let mut values = reserve(plan.blocks)?;
        values.extend(source.into_iter().map(|(_, block)| block));
        values.resize(plan.blocks, ZERO);
        let mut transposed = reserve(plan.max_front)?;
        let mut start = 0;
        let mut targets = plan.updates.iter();
        for column in &plan.columns {
            let index = column.diagonal as usize;
            let l = cholesky(values[index])?;
            values[index] = l;
            let front = &plan.front[start..column.end];
            transposed.clear();
            for edge in front {
                let index = edge.slot as usize;
                let raw = if column.body < edge.body { transpose(values[index]) } else { values[index] };
                let scaled = raw.map(|row| forward(&l, row));
                if scaled.iter().flatten().any(|value| !value.is_finite()) {
                    return Err("nonfinite multiplier");
                }
                values[index] = scaled;
                transposed.push(transpose(scaled));
            }
            for (i, edge) in front.iter().enumerate() {
                let left = values[edge.slot as usize];
                for right in &transposed[i..] {
                    let target = *targets.next().expect("symbolic update target") as usize;
                    for (row, left_row) in values[target].iter_mut().zip(&left) {
                        let mut correction = [0.0; 6];
                        for (&left_lane, right_row) in left_row.iter().zip(right) {
                            for (sum, &right_lane) in correction.iter_mut().zip(right_row) {
                                *sum += left_lane * right_lane;
                            }
                        }
                        for (entry, correction) in row.iter_mut().zip(correction) {
                            *entry -= correction;
                        }
                    }
                }
            }
            start = column.end;
        }
        debug_assert!(targets.next().is_none());
        let stats = Stats { original_edges: plan.original_edges, fill_edges: plan.fill_edges,
                            max_front: plan.max_front, charged_bytes, max_asymmetry };
        cache.plan = Some(Arc::clone(&plan));
        Ok((Self { plan, values }, stats, hit))
    }

    pub(super) fn solve(&self, rhs: &[BodyVector]) -> Result<Vec<BodyVector>> {
        if rhs.len() != self.plan.columns.len() || rhs.iter().flatten().any(|value| !value.is_finite()) {
            return Err("invalid RHS");
        }
        let mut x = reserve(rhs.len())?;
        x.extend_from_slice(rhs);
        let mut start = 0;
        for column in &self.plan.columns {
            let y = forward(&self.values[column.diagonal as usize], x[column.body]);
            x[column.body] = y;
            for edge in &self.plan.front[start..column.end] {
                let w = &self.values[edge.slot as usize];
                for (entry, row) in x[edge.body].iter_mut().zip(w) {
                    *entry -= dot(row, &y);
                }
            }
            start = column.end;
        }
        for (i, column) in self.plan.columns.iter().enumerate().rev() {
            let start = i.checked_sub(1).map_or(0, |previous| self.plan.columns[previous].end);
            let mut y = x[column.body];
            for edge in &self.plan.front[start..column.end] {
                let w = &self.values[edge.slot as usize];
                let z = &x[edge.body];
                for j in 0..6 {
                    let correction: f64 = (0..6).map(|k| w[k][j] * z[k]).sum();
                    y[j] -= correction;
                }
            }
            x[column.body] = backward(&self.values[column.diagonal as usize], y);
        }
        if x.iter().flatten().any(|value| !value.is_finite()) {
            return Err("nonfinite solve");
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Factor;

    const BUDGET: Budget = Budget { bytes: 1 << 20, fill_blocks: 64 };

    fn diagonal(value: f32) -> [[f32; 6]; 6] {
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { value } else { 0.0 }))
    }

    #[test]
    fn solves_cyclic_nonscalar_blocks_and_rebuilds_values_on_symbolic_hit() {
        let mut matrix = BTreeMap::new();
        let off = std::array::from_fn(|i| std::array::from_fn(|j| (1 + 2 * i + j) as f32 / 100.0));
        for body in 0..4 {
            matrix.insert((body, body), diagonal(4.0));
            let other = (body + 1) % 4;
            matrix.insert((body, other), off);
            matrix.insert((other, body), std::array::from_fn(|i| std::array::from_fn(|j| off[j][i])));
        }
        let rhs = std::array::from_fn::<_, 4, _>(|a| std::array::from_fn(|b| (a * 6 + b) as f64));
        let mut cache = Cache::default();
        let (first, hit) = Factor::build_compact(4, &matrix, BUDGET, 100, &mut cache).unwrap();
        assert!(!hit);
        assert_eq!(first.stats.fill_edges, 1);
        let no_fill = Budget { fill_blocks: 0, ..BUDGET };
        assert_eq!(Factor::build_compact(4, &matrix, no_fill, 100, &mut Cache::default()).err(),
                   Some("fill budget"));
        matrix.insert((2, 2), diagonal(6.0));
        let (fresh, hit) = Factor::build_compact(4, &matrix, BUDGET, 100, &mut cache).unwrap();
        assert!(hit);
        let actual = fresh.solve(&rhs).unwrap();
        let expected = Factor::build(4, &matrix, BUDGET).unwrap().solve(&rhs).unwrap();
        assert!(actual.iter().flatten().zip(expected.iter().flatten())
            .all(|(x, y)| (x - y).abs() <= 1e-12));
        assert_ne!(first.solve(&rhs).unwrap(), actual);
        assert_eq!(actual, fresh.solve(&rhs).unwrap());
        matrix.insert((1, 3), diagonal(0.2));
        assert!(!Factor::build_compact(4, &matrix, BUDGET, 100, &mut cache).unwrap().1);
    }

    #[test]
    fn refuses_bad_pivot_missing_diagonal_and_exhausted_budgets() {
        let mut cache = Cache::default();
        let mut matrix = BTreeMap::from([((0, 0), diagonal(-1.0))]);
        assert_eq!(Factor::build_compact(1, &matrix, BUDGET, 100, &mut cache).err(),
                   Some("nonpositive or nonfinite pivot"));
        matrix.insert((0, 0), diagonal(4.0));
        assert_eq!(Factor::build_compact(2, &matrix, BUDGET, 100, &mut cache).err(), Some("missing diagonal"));
        matrix.insert((1, 1), diagonal(4.0));
        matrix.insert((0, 1), diagonal(0.3));
        assert!(Factor::build_compact(2, &matrix, BUDGET, 1, &mut cache).is_ok());
        assert_eq!(Factor::build_compact(2, &matrix, BUDGET, 0, &mut cache).err(), Some("work budget"));
        assert_eq!(Factor::build_compact(2, &matrix, Budget { bytes: 1, ..BUDGET }, 100, &mut cache).err(),
                   Some("workspace budget"));
    }
}
