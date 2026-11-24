use ppf_cts_compute::{Device,Fault,Handle};
use super::{operator::Operator,pdrd::Reduction,joints,body_block_projection::{self,Block,Joint}};
#[path = "compact_joint_export.rs"]
mod compact_joint_export;
fn bad(s:&str)->Fault{Fault::Platform{call:"direct body projection",detail:s.into()}}
fn try_vec<T>(n: usize) -> Result<Vec<T>, Fault> {
    let mut data = Vec::new();
    data.try_reserve_exact(n).map_err(|_| bad("host scratch allocation"))?;
    Ok(data)
}
fn read<T:Copy+Default>(d:&mut impl Device,h:Handle,n:usize)->Result<Vec<T>,Fault>{
    if n > h.size as usize { return Err(bad("read extent")); }
    let bytes = n.checked_mul(std::mem::size_of::<T>()).ok_or_else(||bad("extent"))?;
    let mut v = try_vec(n)?;
    v.resize(n, T::default());
    if bytes > 0 {
        let b=unsafe{std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(),bytes)};
        d.read(h,0,b)?;
    }
    Ok(v)
}

/// This is an optional preconditioner budget, not a cap on the physical scene.
const HOST_SPARSE_BUDGET_BYTES: usize = 256 * 1024 * 1024;
#[derive(Clone, Copy)]
pub(super) struct HostSparsePlan {
    pub blocks: usize,
    pub pairs: usize,
}
fn host_plan(rows: usize, blocks: usize, tets: usize, joint_records: usize,
    bodies: usize, free: usize, roots: usize, budget: usize,
) -> Option<HostSparsePlan> {
    // Include input CSR/transpose/diagonal mirrors and raw Block stream,
    // simultaneous material+legacy export buffers and the compact joint copy,
    // and both working/final forest vectors. Per actual directed pair we also
    // reserve for both 6x6 maps, unordered edges, candidates and tree scratch.
    let base = (1024usize * 1024)
        .checked_add(rows.checked_mul(128)?)?
        .checked_add(blocks.checked_mul(256)?)?
        .checked_add(tets.checked_mul(32)?)?
        .checked_add(joint_records.checked_mul(12 * 1024)?)?
        .checked_add(bodies.checked_mul(512)?)?
        .checked_add(free.checked_mul(4096)?)?
        .checked_add(roots.checked_mul(128)?)?;
    let pairs = budget.checked_sub(base)? / 1536;
    (pairs >= free).then_some(HostSparsePlan { blocks, pairs })
}
pub(super) fn host_sparse_plan(op: &Operator<'_>, joints: joints::View,
    bodies: usize, free: usize, roots: usize,
) -> Option<HostSparsePlan> {
    let limit = std::env::var("PPF_BODY_TREE_HOST_SPARSE_BUDGET_BYTES").ok()
        .and_then(|v| v.parse::<usize>().ok()).map_or(HOST_SPARSE_BUDGET_BYTES,
            |requested| requested.min(HOST_SPARSE_BUDGET_BYTES));
    host_sparse_plan_at_limit(op, joints, bodies, free, roots, limit)
}

pub(super) fn host_sparse_plan_at_limit(op: &Operator<'_>, joints: joints::View,
    bodies: usize, free: usize, roots: usize, limit: usize,
) -> Option<HostSparsePlan> {
    let rows = op.rows() as usize;
    let fixed = match op.fixed_pattern {
        Some(pattern) => (*pattern.offset.last()? as usize)
            .checked_add(*pattern.transpose_offset.last()? as usize)?,
        None => (op.fixed.index.size as usize)
            .checked_add(op.fixed.transpose_pair.size as usize / 2)?,
    };
    let dynamic = match op.dynamic {
        Some(view) => (view.index.size as usize).checked_add(view.transpose_index.size as usize)?,
        None => 0,
    };
    let records = (joints.count as usize).checked_mul(2)?
        .checked_add(joints.control_count as usize)?
        .checked_add(joints.extension_count as usize)?
        .checked_add(joints.soft.map_or(0, |soft| soft.count as usize))?;
    host_plan(rows, rows.checked_add(fixed)?.checked_add(dynamic)?,
        op.tet_factor.map_or(0, |tet| tet.capture_snapshot().tets as usize),
        records, bodies, free, roots, limit)
}

#[cfg(test)]
mod host_budget_tests {
    use super::*;
    #[test]
    fn host_sparse_budget_uses_live_work_instead_of_dense_extent() {
        for bodies in [192, 768] {
            let rows = 4 * bodies;
            let blocks = 12 * rows;
            let joint_records = 4 * (bodies - 1);
            assert!(host_plan(rows, blocks, 0, joint_records, bodies, bodies, 0, 64).is_none());
            let plan = host_plan(rows, blocks, 0, joint_records, bodies, bodies, 0,
                HOST_SPARSE_BUDGET_BYTES).expect("sparse scene fits");
            assert!(plan.pairs >= 3 * bodies);

            let block = |value| {
                let mut matrix = [[0.0; 6]; 6];
                for (index, row) in matrix.iter_mut().enumerate() { row[index] = value; }
                matrix
            };
            let mut directed = std::collections::BTreeMap::new();
            for body in 0..bodies {
                directed.insert((body, body), block(4.0));
                if body > 0 {
                    directed.insert((body, body - 1), block(-1.0));
                    directed.insert((body - 1, body), block(-1.0));
                }
            }
            assert!(directed.len() <= plan.pairs);
            let matrix = body_block_projection::SparseBodyMatrix {
                body_count: bodies, free: (0..bodies).collect(), blocks: directed,
            };
            let forest = super::super::shadow_tree::FreeForest::from_sparse_weighted(&matrix, bodies)
                .expect("factor within budget");
            assert_eq!(forest.omitted_cycles, 0);
            let rhs: Vec<_> = (0..6*bodies)
                .map(|lane| if lane < 6 || lane >= 6*(bodies - 1) {3.0} else {2.0})
                .collect();
            let solution = forest.apply(&rhs).expect("solve within budget");
            assert!(solution.iter().all(|value| (value - 1.0).abs() < 1e-11));
        }
        assert!(host_plan(1, usize::MAX, 0, 0, 1, 1, 0, HOST_SPARSE_BUDGET_BYTES).is_none());
    }
}
fn csr(d:&mut impl Device,rows:usize,o:Handle,i:Handle,v:Handle,to:Handle,ti:Handle,tv:Option<Handle>,out:&mut Vec<Block>)->Result<(),Fault>{
 let offsets=read::<u32>(d,o,rows+1)?;if offsets.windows(2).any(|w|w[0]>w[1]){return Err(bad("CSR offsets"));}let n=offsets[rows] as usize;let index=read::<u32>(d,i,n)?;let value=read::<f32>(d,v,n.checked_mul(9).ok_or_else(||bad("extent"))?)?;
 for r in 0..rows{for k in offsets[r] as usize..offsets[r+1] as usize{let c=index[k] as usize;if c>=rows{return Err(bad("CSR index"));}let mut a=[0.;9];a.copy_from_slice(&value[9*k..9*k+9]);out.push(Block{row:r,col:c,value:a});}}
 let offsets=read::<u32>(d,to,rows+1)?;if offsets.windows(2).any(|w|w[0]>w[1]){return Err(bad("transpose offsets"));}let n=offsets[rows] as usize;
 let (src,slots):(Vec<u32>,Vec<u32>)=if let Some(tv)=tv{(read(d,ti,n)?,read(d,tv,n)?)}else{let a=read::<u32>(d,ti,2*n)?;(a.iter().step_by(2).copied().collect(),a.iter().skip(1).step_by(2).copied().collect())};
 for r in 0..rows{for k in offsets[r] as usize..offsets[r+1] as usize{let c=src[k] as usize;let s=slots[k] as usize;if c>=rows||s>=value.len()/9{return Err(bad("transpose index"));}let mut a=[0.;9];for x in 0..3{for y in 0..3{a[3*y+x]=value[9*s+3*x+y];}}out.push(Block{row:r,col:c,value:a});}}Ok(())
}
fn fixed_csr(d:&mut impl Device,rows:usize,p:super::fixedcsr::FixedPattern<'_>,v:Handle,out:&mut Vec<Block>)->Result<(),Fault>{
 if p.rows as usize!=rows||p.offset.len()!=rows+1||p.transpose_offset.len()!=rows+1{return Err(bad("CSR extent"));}
 let offsets=p.offset;if offsets.windows(2).any(|w|w[0]>w[1]){return Err(bad("CSR offsets"));}let n=offsets[rows] as usize;
 if p.index.len()<n{return Err(bad("CSR extent"));}let value=read::<f32>(d,v,n.checked_mul(9).ok_or_else(||bad("extent"))?)?;
 for r in 0..rows{for k in offsets[r] as usize..offsets[r+1] as usize{let c=p.index[k] as usize;if c>=rows{return Err(bad("CSR index"));}let mut a=[0.;9];a.copy_from_slice(&value[9*k..9*k+9]);out.push(Block{row:r,col:c,value:a});}}
 let offsets=p.transpose_offset;if offsets.windows(2).any(|w|w[0]>w[1]){return Err(bad("transpose offsets"));}let n=offsets[rows] as usize;
 if p.transpose_pair.len()<n.checked_mul(2).ok_or_else(||bad("extent"))?{return Err(bad("transpose extent"));}
 for r in 0..rows{for k in offsets[r] as usize..offsets[r+1] as usize{let c=p.transpose_pair[2*k] as usize;let s=p.transpose_pair[2*k+1] as usize;if c>=rows||s>=value.len()/9{return Err(bad("transpose index"));}let mut a=[0.;9];for x in 0..3{for y in 0..3{a[3*y+x]=value[9*s+3*x+y];}}out.push(Block{row:r,col:c,value:a});}}Ok(())
}
pub unsafe fn capture(d:&mut impl Device,op:&Operator,reduction:&mut Reduction<'_>,joints:joints::View,free:&[usize])->Result<Vec<f32>,Fault>{
 let map=&*reduction.map;let rows=map.nrow;if op.rows() as usize!=rows||free.len()%6!=0{return Err(bad("layout"));}let mut fb=Vec::new();for c in free.chunks_exact(6){let b=c[0]/6;if c.iter().enumerate().any(|(i,&x)|x!=6*b+i){return Err(bad("partial root"));}fb.push(b);}
 let prot=read::<f32>(d,reduction.staged.prot.handle(),3*rows)?;
 let val=if let Some(t)=op.tet_factor{let s=t.capture_snapshot();let verts=read::<u32>(d,s.tet,4*s.tets as usize)?;if verts.iter().any(|&v|v as usize>=rows||map.vbody[v as usize]!=u32::MAX){return Err(bad("rigid tet unsupported"));}s.non_tet}else{op.fixed.value};
 let mut blocks=Vec::new();if let Some(p)=op.fixed_pattern{fixed_csr(d,rows,p,val,&mut blocks)?;}else{csr(d,rows,op.fixed.offset,op.fixed.index,val,op.fixed.transpose_offset,op.fixed.transpose_pair,None,&mut blocks)?;}
 if let Some(a)=op.dynamic{csr(d,rows,a.offset,a.index,a.value,a.transpose_offset,a.transpose_index,Some(a.transpose_value),&mut blocks)?;}
 let diag=read::<f32>(d,op.diagonal,9*rows)?;for r in 0..rows{let mut a=[0.;9];a.copy_from_slice(&diag[9*r..9*r+9]);blocks.push(Block{row:r,col:r,value:a});}
 let mut material=super::reduced_schwarz::FrozenJointBlocks::default();let mut legacy=super::legacy_schwarz::LegacyJointBlocks::default();
 let result=(|| -> Result<Vec<f32>,Fault> {let v=if let Some(s)=joints.soft{material.refresh(d,s)?}else{material.empty(d)?};let v=legacy.refresh(d,joints,v)?;let n=v.records as usize;
 let counts=read::<u32>(d,v.scalar_count,n)?;let ids=read::<u32>(d,v.scalar_ids,24*n)?;let h=read::<f32>(d,v.local_hessian,576*n)?;let mut all_ids=Vec::new();let mut all_h=Vec::new();for k in 0..n{let q=counts[k] as usize;if q>24{return Err(bad("joint count"));}all_ids.push(ids[24*k..24*k+q].to_vec());let mut a=Vec::new();for i in 0..q{a.extend_from_slice(&h[576*k+24*i..576*k+24*i+q]);}all_h.push(a);}
 let js:Vec<_>=all_ids.iter().zip(&all_h).map(|(ids,h)|Joint{ids,h}).collect();eprintln!("PPF_TREE_DIRECT_SOURCES rows={} blocks={} joints={} tet_factor={}",rows,blocks.len(),n,op.has_tet_factor());
 body_block_projection::assemble(&map.vbody,&prot,map.n_bodies,&fb,&blocks,map.body_base,&js).map_err(bad)
 })();let a=legacy.release(d);let b=material.release(d);match result {Err(e)=>Err(e),Ok(v)=>{a?;b?;Ok(v)}}
}

pub unsafe fn capture_sparse(d:&mut impl Device,op:&Operator,reduction:&mut Reduction<'_>,joints:joints::View,free:&[usize],plan:Option<HostSparsePlan>)->Result<body_block_projection::SparseBodyMatrix,Fault>{
    capture_sparse_inner(d, op, reduction, joints, free, plan, true)
}

pub(super) unsafe fn capture_sparse_quiet(d: &mut impl Device, op: &Operator, reduction: &mut Reduction<'_>,
    joints: joints::View, free: &[usize], plan: HostSparsePlan,
) -> Result<body_block_projection::SparseBodyMatrix, Fault> {
    capture_sparse_inner(d, op, reduction, joints, free, Some(plan), false)
}

unsafe fn capture_sparse_inner(d: &mut impl Device, op: &Operator, reduction: &mut Reduction<'_>,
    joints: joints::View, free: &[usize], plan: Option<HostSparsePlan>, log_sources: bool,
) -> Result<body_block_projection::SparseBodyMatrix, Fault> {
 let map=&*reduction.map;let rows=map.nrow;if op.rows() as usize!=rows||free.len()%6!=0{return Err(bad("layout"));}let mut fb=try_vec(free.len()/6)?;for c in free.chunks_exact(6){let b=c[0]/6;if c.iter().enumerate().any(|(i,&x)|x!=6*b+i){return Err(bad("partial root"));}fb.push(b);}
 let prot=read::<f32>(d,reduction.staged.prot.handle(),3*rows)?;
 let val=if let Some(t)=op.tet_factor{let s=t.capture_snapshot();let verts=read::<u32>(d,s.tet,4*s.tets as usize)?;if verts.iter().any(|&v|v as usize>=rows||map.vbody[v as usize]!=u32::MAX){return Err(bad("rigid tet unsupported"));}s.non_tet}else{op.fixed.value};
 let mut blocks=try_vec(plan.map_or(0, |value| value.blocks))?;if let Some(p)=op.fixed_pattern{fixed_csr(d,rows,p,val,&mut blocks)?;}else{csr(d,rows,op.fixed.offset,op.fixed.index,val,op.fixed.transpose_offset,op.fixed.transpose_pair,None,&mut blocks)?;}
 if let Some(a)=op.dynamic{csr(d,rows,a.offset,a.index,a.value,a.transpose_offset,a.transpose_index,Some(a.transpose_value),&mut blocks)?;}
 let diag=read::<f32>(d,op.diagonal,9*rows)?;for r in 0..rows{let mut a=[0.;9];a.copy_from_slice(&diag[9*r..9*r+9]);blocks.push(Block{row:r,col:r,value:a});}
 let mut material=super::reduced_schwarz::FrozenJointBlocks::default();let mut legacy=super::legacy_schwarz::LegacyJointBlocks::default();
 let result=(|| -> Result<body_block_projection::SparseBodyMatrix,Fault> {let v=if let Some(s)=joints.soft{material.refresh(d,s)?}else{material.empty(d)?};let v=legacy.refresh(d,joints,v)?;let n=v.records as usize;
 let counts = read::<u32>(d, v.scalar_count, n)?;
 let ids = read::<u32>(d, v.scalar_ids, compact_joint_export::WIDTH * n)?;
 if let Some(h) = compact_joint_export::read_hessian(d, v.local_hessian, &counts)? {
     let js = compact_joint_export::views(&counts, &ids, &h)?;
     if log_sources { eprintln!("PPF_TREE_DIRECT_SOURCES rows={} blocks={} joints={} tet_factor={}",rows,blocks.len(),n,op.has_tet_factor()); }
     return body_block_projection::assemble_sparse(&map.vbody,&prot,map.n_bodies,&fb,&blocks,
         map.body_base,&js,plan.map(|value| value.pairs)).map_err(bad);
 }
 let h=compact_joint_export::read_padded(d,v.local_hessian,n)?;let mut all_ids=try_vec(n)?;let mut all_h=try_vec(n)?;for k in 0..n{let q=counts[k] as usize;if q>24{return Err(bad("joint count"));}let mut record_ids=try_vec(q)?;record_ids.extend_from_slice(&ids[24*k..24*k+q]);all_ids.push(record_ids);let mut a=try_vec(q*q)?;for i in 0..q{a.extend_from_slice(&h[576*k+24*i..576*k+24*i+q]);}all_h.push(a);}
 let mut js=try_vec(n)?;js.extend(all_ids.iter().zip(&all_h).map(|(ids,h)|Joint{ids,h}));
 if log_sources { eprintln!("PPF_TREE_DIRECT_SOURCES rows={} blocks={} joints={} tet_factor={}",rows,blocks.len(),n,op.has_tet_factor()); }
 body_block_projection::assemble_sparse(&map.vbody,&prot,map.n_bodies,&fb,&blocks,map.body_base,&js,plan.map(|value| value.pairs)).map_err(bad)
 })();
 let legacy_cleanup=legacy.release(d);
 let material_cleanup=material.release(d);
 if plan.is_some() {
     // A refused experiment may fall back only after both exporters have
     // released their scratch. Preserve a cleanup fault over the refusal.
     legacy_cleanup?;
     material_cleanup?;
     result
 } else {
     match result {Err(e)=>Err(e),Ok(v)=>{legacy_cleanup?;material_cleanup?;Ok(v)}}
 }
}


