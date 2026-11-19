//! Directed full-row 3x3 CSR for opt-in operator diagnostics.
use std::collections::BTreeMap;
const MAX_BYTES: usize = 512 * 1024 * 1024;
#[derive(Clone,Copy,Debug)]
pub(super) struct DirectedBlock { pub row:u32, pub col:u32, pub value:[f32;9] }
pub(super) struct Packed { pub rows:u32, pub index:Vec<u32>, pub offset:Vec<u32>, pub value:Vec<f32>, pub transpose_offset:Vec<u32> }
/// Duplicate blocks are accumulated in f64 in input order, rounded once to f32.
pub(super) fn pack(rows:u32, blocks:&[DirectedBlock])->Result<Packed,&'static str>{
 let r=rows as usize; let dim=r.checked_mul(3).ok_or("extent")?; if dim>u32::MAX as usize{return Err("extent")}
 let budget=blocks.len().checked_mul(128).and_then(|x|r.checked_add(1).and_then(|y|y.checked_mul(8)).and_then(|y|x.checked_add(y))).ok_or("extent")?; if budget>MAX_BYTES{return Err("memory limit")}
 let mut map=BTreeMap::<(u32,u32),[f64;9]>::new();
 for b in blocks {if b.row>=rows||b.col>=rows{return Err("index")}if b.value.iter().any(|v|!v.is_finite()){return Err("nonfinite")} let a=map.entry((b.row,b.col)).or_insert([0.;9]);for i in 0..9 {a[i]+=b.value[i] as f64;if !a[i].is_finite(){return Err("nonfinite")}}}
 let n=map.len();if n>u32::MAX as usize || n.checked_mul(9).ok_or("extent")?>u32::MAX as usize || n.checked_mul(36).ok_or("extent")?>u32::MAX as usize{return Err("extent")}
 let mut out=Packed{rows,index:Vec::with_capacity(n),offset:vec![0;r+1],value:Vec::with_capacity(n*9),transpose_offset:vec![0;r+1]};
 for ((row,col),a) in map {out.index.push(col);out.offset[row as usize+1]+=1;for v in a {let x=v as f32;if !x.is_finite(){return Err("nonfinite")}out.value.push(x);}}
 for i in 0..r {out.offset[i+1]=out.offset[i+1].checked_add(out.offset[i]).ok_or("extent")?;}
 Ok(out)
}
#[cfg(test)] mod tests {
 use super::*;
 fn b(r:u32,c:u32,v:[f32;9])->DirectedBlock{DirectedBlock{row:r,col:c,value:v}}
 fn apply(p:&Packed,x:&[f32])->Vec<f32>{let mut y=vec![0.;3*p.rows as usize];for r in 0..p.rows as usize{for k in p.offset[r] as usize..p.offset[r+1] as usize{for i in 0..3{for j in 0..3{y[3*r+i]+=p.value[9*k+3*j+i]*x[3*p.index[k] as usize+j];}}}}y}
 #[test] fn directed_and_duplicate(){let a=[1.,2.,3.,4.,5.,6.,7.,8.,9.];let z=[0.;9];let blocks=[b(0,0,a),b(0,1,[1.;9]),b(1,0,[2.;9]),b(1,1,a),b(0,1,[3.;9])];let p=pack(2,&blocks).unwrap();let x=[1.,2.,3.,4.,5.,6.];let y=apply(&p,&x);let expect=[30.+4.*15.,36.+4.*15.,42.+4.*15.,2.*6.+66.,2.*6.+81.,2.*6.+96.];assert_eq!(y,expect);assert_eq!(p.offset,[0,2,4]);assert_eq!(p.transpose_offset,[0,0,0]);let _=z;}
 #[test] fn mixed_sectors_empty(){let one=[1.,0.,0.,0.,1.,0.,0.,0.,1.];let p=pack(4,&[b(0,0,one),b(0,2,one),b(2,0,one),b(2,2,one)]).unwrap();assert_eq!(p.offset,[0,2,2,4,4]);assert_eq!(apply(&p,&[1.;12]),[2.,2.,2.,0.,0.,0.,2.,2.,2.,0.,0.,0.]);let q=pack(3,&[]).unwrap();assert_eq!(q.offset,[0,0,0,0]);assert!(q.value.is_empty());}
 #[test] fn rejects(){assert!(pack(1,&[b(1,0,[0.;9])]).is_err());assert!(pack(1,&[b(0,0,[f32::NAN;9])]).is_err());assert!(pack(1,&[b(0,0,[f32::MAX;9]),b(0,0,[f32::MAX;9])]).is_err());assert!(pack(u32::MAX,&[]).is_err());}
}

