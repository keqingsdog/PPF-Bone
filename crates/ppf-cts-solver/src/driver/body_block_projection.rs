//! Host assembly of a body principal block from full-space 3x3 blocks and reduced joint Hessians.
//! Values are column-major 3x3; result is column-major in the supplied free-body order.
#[path = "body_block_compact.rs"]
mod compact;

#[derive(Clone, Copy)]
pub struct Block { pub row: usize, pub col: usize, pub value: [f32; 9] }
pub struct Joint<'a> { pub ids: &'a [u32], pub h: &'a [f32] }

pub fn assemble(vbody: &[u32], prot: &[f32], bodies: usize, free: &[usize],
                blocks: &[Block], body_base: usize, joints: &[Joint<'_>]) -> Result<Vec<f32>, &'static str> {
    if prot.len()!=vbody.len().checked_mul(3).ok_or("extent")? {return Err("prot extent");}
    let mut slots=vec![None;bodies];
    for (i,&b) in free.iter().enumerate() {if b>=bodies || slots[b].replace(i).is_some(){return Err("free body mapping");}}
    let dim=free.len().checked_mul(6).ok_or("extent")?;
    let mut out=vec![0f64;dim.checked_mul(dim).ok_or("extent")?];
    let basis=|v:usize| -> Result<Option<(usize,[[f64;6];3])>, &'static str> {
        let b=*vbody.get(v).ok_or("vertex index")?;
        if b==u32::MAX{return Ok(None)}
        let b=b as usize;if b>=bodies{return Err("body index")}
        let Some(s)=slots[b] else{return Ok(None)};
        let p=&prot[3*v..3*v+3];if p.iter().any(|x|!x.is_finite()){return Err("nonfinite prot")}
        let (x,y,z)=(p[0] as f64,p[1] as f64,p[2] as f64);
        Ok(Some((s,[[1.,0.,0.,0.,z,-y],[0.,1.,0.,-z,0.,x],[0.,0.,1.,y,-x,0.]])))
    };
    for block in blocks {
        if block.value.iter().any(|v|!v.is_finite()){return Err("nonfinite block")}
        let (Some((a,ja)),Some((b,jb)))=(basis(block.row)?,basis(block.col)?) else{continue};
        // Rows containing structurally nonzero entries in each rotation column.
        const ROT_ROWS: [[usize; 2]; 3] = [[1, 2], [0, 2], [0, 1]];
        for i in 0..6 {for j in 0..6 {
            let mut v=0.;
            if i<3 && j<3 {
                v=block.value[3*j+i] as f64;
            } else if i<3 {
                for &c in &ROT_ROWS[j-3] {v+=(block.value[3*c+i] as f64)*jb[c][j];}
            } else if j<3 {
                for &r in &ROT_ROWS[i-3] {v+=ja[r][i]*(block.value[3*j+r] as f64);}
            } else {
                for &r in &ROT_ROWS[i-3] {for &c in &ROT_ROWS[j-3] {
                    v+=(ja[r][i]*(block.value[3*c+r] as f64))*jb[c][j];
                }}
            }
            out[(6*b+j)*dim+6*a+i]+=v;
        }}
    }
    for joint in joints {
        let n=joint.ids.len();if n>24||joint.h.len()!=n*n{return Err("joint extent")}
        for i in 0..n {for j in 0..n {
            let h=joint.h[n*i+j];if !h.is_finite(){return Err("nonfinite joint")}
            let (a,b)=(joint.ids[i] as usize,joint.ids[j] as usize);
            if a<body_base||b<body_base||a==u32::MAX as usize||b==u32::MAX as usize {continue}
            let (a,b)=(a-body_base,b-body_base);if a>=6*bodies||b>=6*bodies{return Err("joint index")}
            if let (Some(x),Some(y))=(slots[a/6],slots[b/6]) {out[(6*y+b%6)*dim+6*x+a%6]+=h as f64;}
        }}
    }
    out.into_iter().map(|v|{let x=v as f32;if x.is_finite(){Ok(x)}else{Err("nonfinite result")}}).collect()
}

pub struct SparseBodyMatrix { pub body_count:usize, pub free:Vec<usize>, pub blocks:std::collections::BTreeMap<(usize,usize),[[f32;6];6]> }

pub fn assemble_sparse(vbody: &[u32], prot: &[f32], bodies: usize, free: &[usize],
                blocks: &[Block], body_base: usize, joints: &[Joint<'_>],
                pair_limit: Option<usize>,
) -> Result<SparseBodyMatrix, &'static str> {
    static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static COMPACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let fast = *FAST.get_or_init(|| enabled("PPF_SPARSE_CAPTURE_ARITH"));
    let compact = *COMPACT.get_or_init(|| enabled("PPF_SPARSE_CAPTURE_COMPACT"));
    let trace = *TRACE.get_or_init(|| enabled("PPF_SPARSE_CAPTURE_ARITH_TRACE"))
        && !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed);
    let start = trace.then(std::time::Instant::now);
    let result = if compact {
        compact::assemble(vbody, prot, bodies, free, blocks, body_base, joints, pair_limit)
    } else {
        assemble_sparse_impl(vbody, prot, bodies, free, blocks, body_base, joints, pair_limit, fast)
    };
    if let Some(start) = start {
        let elapsed = start.elapsed();
        let zero = blocks.iter().filter(|block| all_zero(&block.value)).count();
        let joint_values: usize = joints.iter().map(|j| j.h.len()).sum();
        eprintln!(
            "PPF_SPARSE_CAPTURE_ARITH fast={fast} compact={compact} vertices={} blocks={} zero_blocks={zero} joints={} joint_values={joint_values} pairs={:?} elapsed_us={}",
            vbody.len(), blocks.len(), joints.len(), result.as_ref().ok().map(|m| m.blocks.len()), elapsed.as_micros(),
        );
    }
    result
}

fn enabled(key: &str) -> bool {
    std::env::var_os(key).is_some_and(|v| v == "1")
}

#[allow(clippy::too_many_arguments)]
fn assemble_sparse_impl(
    vbody: &[u32], prot: &[f32], bodies: usize, free: &[usize], blocks: &[Block],
    body_base: usize, joints: &[Joint<'_>], pair_limit: Option<usize>, fast: bool,
) -> Result<SparseBodyMatrix, &'static str> {
    if prot.len()!=vbody.len().checked_mul(3).ok_or("extent")? {return Err("prot extent");}
    let mut slots=vec![None;bodies];
    for (i,&b) in free.iter().enumerate() {if b>=bodies || slots[b].replace(i).is_some(){return Err("free body mapping");}}
    free.len().checked_mul(6).ok_or("extent")?;
    let mut out=std::collections::BTreeMap::<(usize,usize),[[f64;6];6]>::new();
    let basis=|v:usize| -> Result<Option<(usize,[[f64;6];3])>, &'static str> {
        let b=*vbody.get(v).ok_or("vertex index")?;
        if b==u32::MAX{return Ok(None)}
        let b=b as usize;if b>=bodies{return Err("body index")}
        let Some(s)=slots[b] else{return Ok(None)};
        let p=&prot[3*v..3*v+3];if p.iter().any(|x|!x.is_finite()){return Err("nonfinite prot")}
        let (x,y,z)=(p[0] as f64,p[1] as f64,p[2] as f64);
        Ok(Some((s,[[1.,0.,0.,0.,z,-y],[0.,1.,0.,-z,0.,x],[0.,0.,1.,y,-x,0.]])))
    };
    for block in blocks {
        if block.value.iter().any(|v|!v.is_finite()){return Err("nonfinite block")}
        let (Some((a,ja)),Some((b,jb)))=(basis(block.row)?,basis(block.col)?) else{continue};
        if pair_limit.is_some_and(|limit| out.len() >= limit && !out.contains_key(&(a,b))) {
            return Err("host sparse pair budget");
        }
        let target=out.entry((a,b)).or_insert([[0.;6];6]);
        if fast {
            // Validate both endpoints and materialize the original key before
            // omitting arithmetic on a zero block.
            if !all_zero(&block.value) {
                add_projected(target, &block.value, &ja, &jb);
            }
            continue;
        }
        // Rows containing structurally nonzero entries in each rotation column.
        const ROT_ROWS: [[usize; 2]; 3] = [[1, 2], [0, 2], [0, 1]];
        for i in 0..6 {for j in 0..6 {
            let mut v=0.;
            if i<3 && j<3 {
                v=block.value[3*j+i] as f64;
            } else if i<3 {
                for &c in &ROT_ROWS[j-3] {v+=(block.value[3*c+i] as f64)*jb[c][j];}
            } else if j<3 {
                for &r in &ROT_ROWS[i-3] {v+=ja[r][i]*(block.value[3*j+r] as f64);}
            } else {
                for &r in &ROT_ROWS[i-3] {for &c in &ROT_ROWS[j-3] {
                    v+=(ja[r][i]*(block.value[3*c+r] as f64))*jb[c][j];
                }}
            }
            target[i][j]+=v;
        }}
    }
    for joint in joints {
        let n=joint.ids.len();if n>24||joint.h.len()!=n*n{return Err("joint extent")}
        if fast {
            add_joint_grouped(&mut out, joint, &slots, body_base, pair_limit)?;
            continue;
        }
        for i in 0..n {for j in 0..n {
            let h=joint.h[n*i+j];if !h.is_finite(){return Err("nonfinite joint")}
            let (a,b)=(joint.ids[i] as usize,joint.ids[j] as usize);
            if a<body_base||b<body_base||a==u32::MAX as usize||b==u32::MAX as usize {continue}
            let (a,b)=(a-body_base,b-body_base);if a>=6*bodies||b>=6*bodies{return Err("joint index")}
            if let (Some(x),Some(y))=(slots[a/6],slots[b/6]) {
                if pair_limit.is_some_and(|limit| out.len() >= limit && !out.contains_key(&(x,y))) {
                    return Err("host sparse pair budget");
                }
                out.entry((x,y)).or_insert([[0.;6];6])[a%6][b%6]+=h as f64;
            }
        }}
    }
    let mut converted=std::collections::BTreeMap::new(); for (key,block) in out {let mut dst=[[0f32;6];6]; for i in 0..6 {for j in 0..6 {dst[i][j]=block[i][j] as f32;if !dst[i][j].is_finite(){return Err("nonfinite result");}}} converted.insert(key,dst);} Ok(SparseBodyMatrix{body_count:bodies,free:free.to_vec(),blocks:converted})
}


fn all_zero(values: &[f32; 9]) -> bool {
    values.iter().all(|&value| value == 0.0)
}

fn add_projected(target: &mut [[f64; 6]; 6], block: &[f32; 9], a: &[[f64; 6]; 3], b: &[[f64; 6]; 3]) {
    const ROT: [[usize; 2]; 3] = [[1, 2], [0, 2], [0, 1]];
    let h = block.map(f64::from);
    for i in 0..3 {
        for j in 0..3 {
            target[i][j] += h[3 * j + i];
            let [c0, c1] = ROT[j];
            let mut tr = 0.0;
            tr += h[3 * c0 + i] * b[c0][j + 3];
            tr += h[3 * c1 + i] * b[c1][j + 3];
            target[i][j + 3] += tr;
            let [r0, r1] = ROT[i];
            let mut rt = 0.0;
            rt += a[r0][i + 3] * h[3 * j + r0];
            rt += a[r1][i + 3] * h[3 * j + r1];
            target[i + 3][j] += rt;
            // Match the original r-outer/c-inner sum, including its signed
            // zero seed and the order of the two factors in every product.
            let mut rr = 0.0;
            rr += (a[r0][i + 3] * h[3 * c0 + r0]) * b[c0][j + 3];
            rr += (a[r0][i + 3] * h[3 * c1 + r0]) * b[c1][j + 3];
            rr += (a[r1][i + 3] * h[3 * c0 + r1]) * b[c0][j + 3];
            rr += (a[r1][i + 3] * h[3 * c1 + r1]) * b[c1][j + 3];
            target[i + 3][j + 3] += rr;
        }
    }
}

#[derive(Clone, Copy)]
enum JointSlot {
    Ignore,
    Invalid,
    Fixed,
    Free(usize, usize),
}

fn add_joint_grouped(
    out: &mut std::collections::BTreeMap<(usize, usize), [[f64; 6]; 6]>,
    joint: &Joint<'_>, slots: &[Option<usize>], body_base: usize, pair_limit: Option<usize>,
) -> Result<(), &'static str> {
    use std::collections::btree_map::Entry;
    use JointSlot::*;
    let n = joint.ids.len();
    let mut mapped = [Ignore; 24];
    for (dest, &raw) in mapped.iter_mut().zip(joint.ids) {
        let index = raw as usize;
        *dest = if raw == u32::MAX || index < body_base {
            Ignore
        } else {
            let index = index - body_base;
            match slots.get(index / 6) {
                Some(Some(slot)) => Free(*slot, index % 6),
                Some(None) => Fixed,
                None => Invalid,
            }
        };
    }
    for (i, &row) in mapped[..n].iter().enumerate() {
        let h = &joint.h[i * n..(i + 1) * n];
        let mut j = 0;
        while j < n {
            let value = h[j];
            if !value.is_finite() { return Err("nonfinite joint"); }
            match (row, mapped[j]) {
                (Ignore, _) | (_, Ignore) => {}
                (Invalid, _) | (_, Invalid) => return Err("joint index"),
                (Free(x, row), Free(y, col)) => {
                    let at_limit = pair_limit.is_some_and(|limit| out.len() >= limit);
                    let block = match out.entry((x, y)) {
                        Entry::Occupied(entry) => entry.into_mut(),
                        Entry::Vacant(_) if at_limit => return Err("host sparse pair budget"),
                        Entry::Vacant(entry) => entry.insert([[0.; 6]; 6]),
                    };
                    block[row][col] += f64::from(value);
                    // Reuse this entry only across consecutive columns of the
                    // same body. Each addition and validation stays in order.
                    while j + 1 < n {
                        let Free(next, col) = mapped[j + 1] else { break; };
                        if next != y { break; }
                        j += 1;
                        let value = h[j];
                        if !value.is_finite() { return Err("nonfinite joint"); }
                        block[row][col] += f64::from(value);
                    }
                }
                _ => {}
            }
            j += 1;
        }
    }
    Ok(())
}

#[cfg(test)] mod tests {
 use super::*;
 #[test] fn projection_matches_dense_oracle_with_roots_and_joints() {
  let vb=[0,0,1,u32::MAX,2];let p=[1.,2.,3., -2.,1.,0.5, 0.,1.,2., 0.,0.,0., 4.,0.,1.];
  let blocks=vec![Block{row:0,col:1,value:[1.,2.,3.,4.,5.,6.,7.,8.,10.]},Block{row:1,col:2,value:[-1.,2.,0.,3.,-2.,1.,5.,4.,1.]},Block{row:2,col:4,value:[1.;9]},Block{row:3,col:0,value:[2.;9]}];
  let ids=[3u32,9,9,15];let h=[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.];
  let got=assemble(&vb,&p,3,&[0,1],&blocks,3,&[Joint{ids:&ids,h:&h}]).unwrap();
  let mut j=vec![vec![0f64;12];15];for v in 0..3 {let b=vb[v] as usize;let (x,y,z)=(p[3*v] as f64,p[3*v+1] as f64,p[3*v+2] as f64);let q=[[1.,0.,0.,0.,z,-y],[0.,1.,0.,-z,0.,x],[0.,0.,1.,y,-x,0.]];for r in 0..3 {for c in 0..6 {j[3*v+r][6*b+c]=q[r][c];}}}
  let mut full=vec![vec![0f64;15];15];for b in &blocks {for r in 0..3 {for c in 0..3 {full[3*b.row+r][3*b.col+c]+=b.value[3*c+r] as f64;}}}
  for a in 0..12 {for b in 0..12 {let mut expected=0.;for r in 0..15 {for c in 0..15 {expected+=j[r][a]*full[r][c]*j[c][b];}}for i in 0..4 {for k in 0..4 {if ids[i] as usize==a+3 && ids[k] as usize==b+3 {expected+=h[4*i+k] as f64;}}}assert!((got[b*12+a] as f64-expected).abs()<1e-5,"{a} {b}");}}
 }
}

#[cfg(test)] mod sparse_tests {
 use super::*;
 #[test] fn sparse_matches_dense_with_roots_fem_and_duplicate_joint_indices() {
  let vb=[0,0,1,u32::MAX,2]; let p=[1.,2.,3.,-2.,1.,0.5,0.,1.,2.,0.,0.,0.,4.,0.,1.];
  let blocks=[Block{row:0,col:1,value:[1.,2.,3.,4.,5.,6.,7.,8.,10.]},Block{row:1,col:2,value:[-1.,2.,0.,3.,-2.,1.,5.,4.,1.]},Block{row:2,col:4,value:[1.;9]},Block{row:3,col:0,value:[2.;9]}];
  let ids=[3u32,9,9,15]; let h=[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]; let joints=[Joint{ids:&ids,h:&h}];
  let dense=assemble(&vb,&p,3,&[1,0],&blocks,3,&joints).unwrap(); let sparse=assemble_sparse(&vb,&p,3,&[1,0],&blocks,3,&joints,None).unwrap();
  for a in 0..2 {for b in 0..2 {for i in 0..6 {for j in 0..6 {let x=sparse.blocks.get(&(a,b)).map_or(0.,|v|v[i][j]);assert_eq!(x.to_bits(),dense[(6*b+j)*12+6*a+i].to_bits());}}}}
 }
 #[test] fn sparse_pair_budget_rejects_a_new_pair_but_allows_duplicates() {
  let block=Block{row:0,col:0,value:[1.;9]};
  let duplicate=Block{row:0,col:0,value:[2.;9]};
  let second=Block{row:1,col:0,value:[1.;9]};
  assert!(assemble_sparse(&[0,1],&[0.;6],2,&[0,1],&[block,duplicate],0,&[],Some(1)).is_ok());
  let limited=assemble_sparse(&[0,1],&[0.;6],2,&[0,1],&[block,second],0,&[],Some(1));
  assert!(matches!(limited,Err("host sparse pair budget")));
 }
 #[test] fn sparse_rejects_nonfinite_even_when_body_is_fixed() {
  let blocks=[Block{row:0,col:1,value:[f32::NAN;9]}];
  assert!(assemble_sparse(&[0,1],&[0.;6],2,&[0],&blocks,0,&[],None).is_err());
 }
}



#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    fn compare(
        owners: &[u32], prot: &[f32], bodies: usize, free: &[usize],
        blocks: &[Block], base: usize, joints: &[Joint<'_>], budget: Option<usize>,
    ) {
        let old = assemble_sparse_impl(owners, prot, bodies, free, blocks, base, joints, budget, false);
        let new = assemble_sparse_impl(owners, prot, bodies, free, blocks, base, joints, budget, true);
        let compact = compact::assemble(owners, prot, bodies, free, blocks, base, joints, budget);
        for candidate in [&new, &compact] {
            match (&old, candidate) {
                (Err(old), Err(new)) => assert_eq!(old, new),
                (Ok(old), Ok(new)) => {
                    assert_eq!(new.body_count, old.body_count);
                    assert_eq!(new.free, old.free);
                    assert_eq!(new.blocks.len(), old.blocks.len());
                    for ((key, a), (new_key, b)) in old.blocks.iter().zip(new.blocks.iter()) {
                        assert_eq!(key, new_key);
                        for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
                            assert_eq!(a.to_bits(), b.to_bits());
                        }
                    }
                }
                (old, new) => panic!("mismatched outcome: {:?} {:?}", old.as_ref().err(), new.as_ref().err()),
            }
        }
    }

    #[test]
    fn differential_finite_repeated_indices_fixed_root_fem_and_zero_keys() {
        let mut state = 73u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as u32
        };
        let owners = [0, 0, 1, 2, 2, 3, u32::MAX, 4];
        let free = [4, 2, 1, 0];
        for _ in 0..128 {
            let prot = std::array::from_fn::<_, 24, _>(|_| (random() as i32 % 109) as f32 / 19.);
            let blocks: Vec<_> = (0..80).map(|i| Block {
                row: (random() % 8) as usize,
                col: (random() % 8) as usize,
                value: if i % 4 == 0 { [if i % 8 == 0 {-0.} else {0.}; 9] }
                       else { std::array::from_fn(|_| (random() as i32 % 311) as f32 / 13.) },
            }).collect();
            let ids = [7, 7, 9, 13, 12, 13, 7, 23, 0, u32::MAX, 31, 34];
            let h = std::array::from_fn::<_, 144, _>(|i| if i % 5 == 0 { -0. } else { (random() as i32 % 571) as f32 / 7. });
            let joints = [Joint { ids: &ids, h: &h }, Joint { ids: &ids, h: &h }];
            for budget in [None, Some(0), Some(3), Some(16)] {
                compare(&owners, &prot, 5, &free, &blocks, 6, &joints, budget);
            }
        }
        let signed = [Block {row: 0, col: 1, value: [-0.; 9]}];
        compare(&[0, 1], &[-0.; 6], 2, &[0, 1], &signed, 0, &[], Some(1));
        let result = assemble_sparse_impl(&[0,1], &[-0.;6], 2, &[0,1], &signed, 0, &[], Some(1), true).unwrap();
        assert!(result.blocks.contains_key(&(0,1)));
        assert!(result.blocks[&(0,1)].iter().flatten().all(|x| x.to_bits() == 0));
    }

    #[test]
    fn differential_preserves_nonfinite_error_order_and_pair_budget() {
        let owners = [0, 1, 2, u32::MAX];
        let free = [1, 0];
        let zero = Block {row: 0, col: 1, value: [0.; 9]};
        let cases = [
            (Block {row: 9, col: 0, value: [f32::NAN; 9]}, [0.;12]),
            (Block {row: 9, col: 0, value: [0.; 9]}, [0.;12]),
            (zero, [f32::NAN;12]),
            (Block {row: 2, col: 3, value: [0.; 9]}, [f32::NAN;12]),
            (Block {row: 0, col: 1, value: [f32::MAX;9]}, [f32::MAX;12]),
        ];
        for (block, prot) in cases {
            compare(&owners, &prot, 3, &free, &[block, zero], 4, &[], Some(1));
        }
        for ids in [[0, u32::MAX, u32::MAX - 1, 5], [4, 4, 4, 10], [0, 0, 0, 0]] {
            for nan_at in [0, 3, 7, 15] {
                let mut h = [0.; 16];
                h[nan_at] = f32::INFINITY;
                for budget in [None, Some(0), Some(1)] {
                    compare(&owners, &[0.;12], 3, &free, &[zero], 4, &[Joint { ids: &ids, h: &h }], budget);
                }
            }
        }
    }
}
