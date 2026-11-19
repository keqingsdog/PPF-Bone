use ppf_cts_compute::{Device,Fault,Handle};
use super::{operator::Operator,pdrd::Reduction,joints,body_block_projection::{self,Block,Joint}};
fn bad(s:&str)->Fault{Fault::Platform{call:"direct body projection",detail:s.into()}}
fn read<T:Copy+Default>(d:&mut impl Device,h:Handle,n:usize)->Result<Vec<T>,Fault>{let mut v=vec![T::default();n];if n>0{let b=unsafe{std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(),n.checked_mul(std::mem::size_of::<T>()).ok_or_else(||bad("extent"))?)};d.read(h,0,b)?;}Ok(v)}
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

pub unsafe fn capture_sparse(d:&mut impl Device,op:&Operator,reduction:&mut Reduction<'_>,joints:joints::View,free:&[usize])->Result<body_block_projection::SparseBodyMatrix,Fault>{
 let map=&*reduction.map;let rows=map.nrow;if op.rows() as usize!=rows||free.len()%6!=0{return Err(bad("layout"));}let mut fb=Vec::new();for c in free.chunks_exact(6){let b=c[0]/6;if c.iter().enumerate().any(|(i,&x)|x!=6*b+i){return Err(bad("partial root"));}fb.push(b);}
 let prot=read::<f32>(d,reduction.staged.prot.handle(),3*rows)?;
 let val=if let Some(t)=op.tet_factor{let s=t.capture_snapshot();let verts=read::<u32>(d,s.tet,4*s.tets as usize)?;if verts.iter().any(|&v|v as usize>=rows||map.vbody[v as usize]!=u32::MAX){return Err(bad("rigid tet unsupported"));}s.non_tet}else{op.fixed.value};
 let mut blocks=Vec::new();if let Some(p)=op.fixed_pattern{fixed_csr(d,rows,p,val,&mut blocks)?;}else{csr(d,rows,op.fixed.offset,op.fixed.index,val,op.fixed.transpose_offset,op.fixed.transpose_pair,None,&mut blocks)?;}
 if let Some(a)=op.dynamic{csr(d,rows,a.offset,a.index,a.value,a.transpose_offset,a.transpose_index,Some(a.transpose_value),&mut blocks)?;}
 let diag=read::<f32>(d,op.diagonal,9*rows)?;for r in 0..rows{let mut a=[0.;9];a.copy_from_slice(&diag[9*r..9*r+9]);blocks.push(Block{row:r,col:r,value:a});}
 let mut material=super::reduced_schwarz::FrozenJointBlocks::default();let mut legacy=super::legacy_schwarz::LegacyJointBlocks::default();
 let result=(|| -> Result<body_block_projection::SparseBodyMatrix,Fault> {let v=if let Some(s)=joints.soft{material.refresh(d,s)?}else{material.empty(d)?};let v=legacy.refresh(d,joints,v)?;let n=v.records as usize;
 let counts=read::<u32>(d,v.scalar_count,n)?;let ids=read::<u32>(d,v.scalar_ids,24*n)?;let h=read::<f32>(d,v.local_hessian,576*n)?;let mut all_ids=Vec::new();let mut all_h=Vec::new();for k in 0..n{let q=counts[k] as usize;if q>24{return Err(bad("joint count"));}all_ids.push(ids[24*k..24*k+q].to_vec());let mut a=Vec::new();for i in 0..q{a.extend_from_slice(&h[576*k+24*i..576*k+24*i+q]);}all_h.push(a);}
 let js:Vec<_>=all_ids.iter().zip(&all_h).map(|(ids,h)|Joint{ids,h}).collect();eprintln!("PPF_TREE_DIRECT_SOURCES rows={} blocks={} joints={} tet_factor={}",rows,blocks.len(),n,op.has_tet_factor());
 body_block_projection::assemble_sparse(&map.vbody,&prot,map.n_bodies,&fb,&blocks,map.body_base,&js).map_err(bad)
 })();let a=legacy.release(d);let b=material.release(d);match result {Err(e)=>Err(e),Ok(v)=>{a?;b?;Ok(v)}}
}


