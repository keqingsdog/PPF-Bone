//! Sparse rigid projection with stable additions and contiguous pair storage.
use super::{add_projected, all_zero, Block, Joint, SparseBodyMatrix};
use std::collections::{BTreeMap, HashMap};

type Matrix = [[f64; 6]; 6];
type Key = (usize, usize);
const ABSENT: usize = usize::MAX;

#[derive(Clone, Copy)]
enum Vertex {
    Omitted,
    InvalidBody,
    InvalidPosition,
    Free(usize, [f32; 3]),
}

impl Vertex {
    fn resolve(self) -> Result<Option<(usize, [f32; 3])>, &'static str> {
        match self {
            Self::Omitted => Ok(None),
            Self::InvalidBody => Err("body index"),
            Self::InvalidPosition => Err("nonfinite prot"),
            Self::Free(slot, position) => Ok(Some((slot, position))),
        }
    }
}

struct Pairs {
    diagonal: Vec<usize>,
    off_diagonal: HashMap<Key, usize>,
    entries: Vec<(Key, Matrix)>,
    limit: Option<usize>,
}

impl Pairs {
    fn index(&mut self, key: Key) -> Result<usize, &'static str> {
        let (row, col) = key;
        if row == col {
            let index = self.diagonal[row];
            if index != ABSENT {
                return Ok(index);
            }
        } else if let Some(&index) = self.off_diagonal.get(&key) {
            return Ok(index);
        }
        let index = self.entries.len();
        if self.limit.is_some_and(|limit| index >= limit) {
            return Err("host sparse pair budget");
        }
        self.entries
            .try_reserve(1)
            .map_err(|_| "workspace allocation")?;
        if row == col {
            self.diagonal[row] = index;
        } else {
            self.off_diagonal
                .try_reserve(1)
                .map_err(|_| "workspace allocation")?;
            self.off_diagonal.insert(key, index);
        }
        self.entries.push((key, [[0.0; 6]; 6]));
        Ok(index)
    }

    fn finish(self, bodies: usize, free: &[usize]) -> Result<SparseBodyMatrix, &'static str> {
        let mut blocks = BTreeMap::new();
        for (key, values) in self.entries {
            let block = values.map(|row| row.map(|value| value as f32));
            if block.iter().flatten().any(|value| !value.is_finite()) {
                return Err("nonfinite result");
            }
            blocks.insert(key, block);
        }
        let mut free_bodies = reserve(free.len())?;
        free_bodies.extend_from_slice(free);
        Ok(SparseBodyMatrix {
            body_count: bodies,
            free: free_bodies,
            blocks,
        })
    }
}

fn reserve<T>(len: usize) -> Result<Vec<T>, &'static str> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| "workspace allocation")?;
    Ok(values)
}

fn basis(position: [f32; 3]) -> [[f64; 6]; 3] {
    let [x, y, z] = position.map(f64::from);
    [
        [1., 0., 0., 0., z, -y],
        [0., 1., 0., -z, 0., x],
        [0., 0., 1., y, -x, 0.],
    ]
}

#[derive(Clone, Copy)]
enum Slot {
    Ignore,
    Invalid,
    Fixed,
    Free(usize, usize),
}

fn add_joint(
    pairs: &mut Pairs,
    joint: &Joint<'_>,
    slots: &[Option<usize>],
    body_base: usize,
    indices: &mut [usize; 24 * 24],
) -> Result<(), &'static str> {
    use Slot::*;
    let count = joint.ids.len();
    if count > 24 || joint.h.len() != count * count {
        return Err("joint extent");
    }
    let mut dofs = [Ignore; 24];
    let mut bodies = [0; 24];
    let mut groups = 0;
    for (&id, dof) in joint.ids.iter().zip(&mut dofs) {
        let index = id as usize;
        if id == u32::MAX || index < body_base {
            continue;
        }
        let index = index - body_base;
        *dof = match slots.get(index / 6) {
            Some(Some(body)) => {
                let group = bodies[..groups]
                    .iter()
                    .position(|old| old == body)
                    .unwrap_or_else(|| {
                        let group = groups;
                        bodies[group] = *body;
                        groups += 1;
                        group
                    });
                Free(group, index % 6)
            }
            Some(None) => Fixed,
            None => Invalid,
        };
    }
    indices[..groups * groups].fill(ABSENT);
    // Consume scalar records in source order. The small per-record table
    // bypasses repeated global lookup without changing duplicate additions.
    for (&row, source) in dofs[..count].iter().zip(joint.h.chunks_exact(count.max(1))) {
        for (&col, &value) in dofs[..count].iter().zip(source) {
            if !value.is_finite() {
                return Err("nonfinite joint");
            }
            match (row, col) {
                (Ignore, _) | (_, Ignore) => {}
                (Invalid, _) | (_, Invalid) => return Err("joint index"),
                (Free(a, i), Free(b, j)) => {
                    let index = &mut indices[a * groups + b];
                    if *index == ABSENT {
                        *index = pairs.index((bodies[a], bodies[b]))?;
                    }
                    pairs.entries[*index].1[i][j] += f64::from(value);
                }
                _ => {}
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assemble(
    vbody: &[u32],
    prot: &[f32],
    bodies: usize,
    free: &[usize],
    blocks: &[Block],
    body_base: usize,
    joints: &[Joint<'_>],
    pair_limit: Option<usize>,
) -> Result<SparseBodyMatrix, &'static str> {
    if prot.len() != vbody.len().checked_mul(3).ok_or("extent")? {
        return Err("prot extent");
    }
    let mut slots = reserve(bodies)?;
    slots.resize(bodies, None);
    for (slot, &body) in free.iter().enumerate() {
        if body >= bodies || slots[body].replace(slot).is_some() {
            return Err("free body mapping");
        }
    }
    free.len().checked_mul(6).ok_or("extent")?;
    let mut vertices = reserve(vbody.len())?;
    for (&body, position) in vbody.iter().zip(prot.chunks_exact(3)) {
        let entry = if body == u32::MAX {
            Vertex::Omitted
        } else {
            match slots.get(body as usize) {
                Some(None) => Vertex::Omitted,
                None => Vertex::InvalidBody,
                Some(Some(slot)) if position.iter().all(|value| value.is_finite()) => {
                    Vertex::Free(*slot, [position[0], position[1], position[2]])
                }
                Some(Some(_)) => Vertex::InvalidPosition,
            }
        };
        vertices.push(entry);
    }
    let mut diagonal = reserve(free.len())?;
    diagonal.resize(free.len(), ABSENT);
    let mut pairs = Pairs {
        diagonal,
        off_diagonal: HashMap::new(),
        entries: Vec::new(),
        limit: pair_limit,
    };
    let vertex = |index| {
        vertices
            .get(index)
            .copied()
            .ok_or("vertex index")
            .and_then(Vertex::resolve)
    };
    for block in blocks {
        let zero = all_zero(&block.value);
        if !zero && block.value.iter().any(|value| !value.is_finite()) {
            return Err("nonfinite block");
        }
        let (Some((a, pa)), Some((b, pb))) = (vertex(block.row)?, vertex(block.col)?) else {
            continue;
        };
        let index = pairs.index((a, b))?;
        if !zero {
            add_projected(
                &mut pairs.entries[index].1,
                &block.value,
                &basis(pa),
                &basis(pb),
            );
        }
    }
    let mut joint_indices = [ABSENT; 24 * 24];
    for joint in joints {
        add_joint(&mut pairs, joint, &slots, body_base, &mut joint_indices)?;
    }
    pairs.finish(bodies, free)
}
