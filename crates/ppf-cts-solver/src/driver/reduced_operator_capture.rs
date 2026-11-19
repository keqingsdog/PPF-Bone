use ppf_cts_compute::{Device,Fault,Handle};
use super::{operator::Operator,pdrd::Reduction,joints,reduced_operator_projection::{self as projection,FullBlock,DirectedBlock}};
const MAX_BLOCKS:usize=4_000_000;
fn bad(s:&str)->Fault{Fault::Platform{call:"reduced operator capture",detail:s.into()}}
fn read<T:Copy+Default>(d:&mut impl Device,h:Handle,n:usize)->Result<Vec<T>,Fault>{let mut v=vec![T::default();n];if n>0{let len=n.checked_mul(std::mem::size_of::<T>()).ok_or_else(||bad("extent"))?;let b=unsafe{std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(),len)};d.read(h,0,b)?;}Ok(v)}
fn emit(map:&super::rigid_map::RigidMap,prot:&[f32],b:FullBlock,out:&mut Vec<DirectedBlock>)->Result<(),Fault>{if out.len()>MAX_BLOCKS-4{return Err(bad("block limit"));}projection::project_block(&map.vbody,&map.cloth_off,prot,map.body_base,map.n_bodies,b,out).map_err(bad)}
fn csr(d:&mut impl Device,map:&super::rigid_map::RigidMap,prot:&[f32],o:Handle,i:Handle,v:Handle,to:Handle,ti:Handle,tv:Option<Handle>,out:&mut Vec<DirectedBlock>)->Result<(),Fault>{
 let rows=map.nrow;let offsets=read::<u32>(d,o,rows+1)?;if offsets.first()!=Some(&0)||offsets.windows(2).any(|w|w[0]>w[1]){return Err(bad("CSR offsets"));}let n=offsets[rows] as usize;if n>MAX_BLOCKS{return Err(bad("CSR block limit"));}let index=read::<u32>(d,i,n)?;let value=read::<f32>(d,v,n.checked_mul(9).ok_or_else(||bad("extent"))?)?;
 for row in 0..rows{for k in offsets[row] as usize..offsets[row+1] as usize{let col=index[k] as usize;if col>=rows{return Err(bad("CSR index"));}let mut x=[0.;9];x.copy_from_slice(&value[9*k..9*k+9]);emit(map,prot,FullBlock{row,col,value:x},out)?;}}
 let offsets=read::<u32>(d,to,rows+1)?;if offsets.first()!=Some(&0)||offsets.windows(2).any(|w|w[0]>w[1]){return Err(bad("transpose offsets"));}let n=offsets[rows] as usize;if n>MAX_BLOCKS{return Err(bad("transpose limit"));}
 let (src,slots):(Vec<u32>,Vec<u32>)=if let Some(tv)=tv{(read(d,ti,n)?,read(d,tv,n)?)}else{let a=read::<u32>(d,ti,n.checked_mul(2).ok_or_else(||bad("extent"))?)?;(a.iter().step_by(2).copied().collect(),a.iter().skip(1).step_by(2).copied().collect())};
 for row in 0..rows{for k in offsets[row] as usize..offsets[row+1] as usize{let col=src[k] as usize;let slot=slots[k] as usize;if col>=rows||slot>=value.len()/9{return Err(bad("transpose index"));}let mut x=[0.;9];for i in 0..3{for j in 0..3{x[3*j+i]=value[9*slot+3*i+j];}}emit(map,prot,FullBlock{row,col,value:x},out)?;}}Ok(())
}
pub unsafe fn capture(d:&mut impl Device,op:&Operator,reduction:&mut Reduction<'_>,j:joints::View)->Result<Vec<DirectedBlock>,Fault>{
 let map=&*reduction.map;let rows=map.nrow;let dim=map.dim;if op.rows() as usize!=rows||map.vbody.len()!=rows||map.cloth_off.len()!=rows||map.body_base%3!=0||map.body_base+6*map.n_bodies!=dim||j.base as usize!=map.body_base||j.dim as usize!=dim||j.bodies as usize!=map.n_bodies||!j.dt.is_finite()||j.dt<=0.{return Err(bad("layout or joint clock"));}
 if let Some(c)=j.curves{if c.dt.to_bits()!=j.dt.to_bits(){return Err(bad("curve dt mismatch"));}}
 if j.count==0 && (j.control_count!=0||j.extension_count!=0){return Err(bad("unsupported control-only legacy export"));}
 for (label,published,epoch,count,factors,stride,offsets) in j.reduced_capture_epochs(){
  if count==0{continue}if epoch==0||(published.size as usize)<count||(factors.size as usize)<count.checked_mul(stride).ok_or_else(||bad("epoch extent"))?{return Err(bad("frozen rigid extent or epoch"));}
  if let Some((h,stride))=offsets{if (h.size as usize)<count.checked_mul(stride).ok_or_else(||bad("epoch extent"))?{return Err(bad("frozen offsets extent"));}}
  if read::<u32>(d,published,count)?.iter().any(|&v|v!=epoch){return Err(bad("frozen rigid stale epoch"));}eprintln!("PPF_REDUCED_EPOCH family={} rows={} epoch={}",label,count,epoch);
 }
 if let Some(s)=j.soft{
  if !s.ready||s.vertices as usize!=rows||s.free_count as usize!=map.n_cloth||s.base!=j.base||s.dim!=j.dim||s.bodies!=j.bodies||s.dt.to_bits()!=j.dt.to_bits(){return Err(bad("soft readiness"));}
  if let Some((legacy,published,epoch,h,idx))=s.reduced_capture_frozen(){let n=s.count as usize;if epoch==0{return Err(bad("soft epoch zero"));}if legacy{eprintln!("PPF_REDUCED_EPOCH family=soft_legacy rows={} epoch={} validation=native_token_probe",n,epoch);}else{if (published.size as usize)<n||(h.size as usize)<n.checked_mul(576).ok_or_else(||bad("soft extent"))?||(idx.size as usize)<n.checked_mul(24).ok_or_else(||bad("soft extent"))?||read::<u32>(d,published,n)?.iter().any(|&v|v!=epoch){return Err(bad("soft stale epoch or extent"));}eprintln!("PPF_REDUCED_EPOCH family=soft_frozen rows={} epoch={}",n,epoch);}}
  if s.resident_cache_ready(){eprintln!("PPF_REDUCED_EPOCH family=soft_resident rows={} epoch={} validation=native_token_probe",s.count,s.hessian_epoch);}
 }
 let prot=read::<f32>(d,reduction.staged.prot.handle(),3*rows)?;let snapshot=op.tet_factor.map(|t|t.capture_snapshot());let fixed=snapshot.map_or(op.fixed.value,|t|t.non_tet);let mut out=Vec::new();
 csr(d,map,&prot,op.fixed.offset,op.fixed.index,fixed,op.fixed.transpose_offset,op.fixed.transpose_pair,None,&mut out)?;
 if let Some(a)=op.dynamic{csr(d,map,&prot,a.offset,a.index,a.value,a.transpose_offset,a.transpose_index,Some(a.transpose_value),&mut out)?;}
 let diag=read::<f32>(d,op.diagonal,9*rows)?;for row in 0..rows{let mut x=[0.;9];x.copy_from_slice(&diag[9*row..9*row+9]);emit(map,&prot,FullBlock{row,col:row,value:x},&mut out)?;}
 if let Some(t)=snapshot{if t.rows as usize!=rows{return Err(bad("tet rows"));}let n=t.tets as usize;if n>MAX_BLOCKS/16{return Err(bad("tet limit"));}let ids=read::<u32>(d,t.tet,n.checked_mul(4).ok_or_else(||bad("extent"))?)?;let f=read::<f32>(d,t.factor,n.checked_mul(81).ok_or_else(||bad("extent"))?)?;let w=read::<f32>(d,t.weight,n.checked_mul(9).ok_or_else(||bad("extent"))?)?;let mask=read::<u32>(d,t.mask,rows)?;for k in 0..n{let verts=[ids[4*k] as usize,ids[4*k+1] as usize,ids[4*k+2] as usize,ids[4*k+3] as usize];for b in projection::tet_blocks(verts,&mask,&f[81*k..81*k+81],&w[9*k..9*k+9]).map_err(bad)?{emit(map,&prot,b,&mut out)?;}}}
 let mut material=super::reduced_schwarz::FrozenJointBlocks::default();let mut legacy=super::legacy_schwarz::LegacyJointBlocks::default();
 let result=(||->Result<(),Fault>{let v=if let Some(soft)=j.soft{material.refresh(d,soft)?}else{material.empty(d)?};let v=legacy.refresh(d,j,v)?;let n=v.records as usize;if n>MAX_BLOCKS/576{return Err(bad("joint record limit"));}let counts=read::<u32>(d,v.scalar_count,n)?;let ids=read::<u32>(d,v.scalar_ids,24*n)?;let h=read::<f32>(d,v.local_hessian,576*n)?;for k in 0..n{let q=counts[k] as usize;if q>24||out.len().checked_add(q*q).map_or(true,|v|v>MAX_BLOCKS){return Err(bad("joint count/limit"));}let mut compact=Vec::with_capacity(q*q);for i in 0..q{compact.extend_from_slice(&h[576*k+24*i..576*k+24*i+q]);}projection::project_joint(&ids[24*k..24*k+q],&compact,dim,&mut out).map_err(bad)?;}eprintln!("PPF_REDUCED_CAPTURE rows={} directed_blocks={} joint_records={} tet_elements={} joint_source=recomputed",dim/3,out.len(),n,snapshot.map_or(0,|x|x.tets));Ok(())})();
 let a=legacy.release(d);let b=material.release(d);match result{Err(e)=>Err(e),Ok(())=>{a?;b?;Ok(out)}}
}
