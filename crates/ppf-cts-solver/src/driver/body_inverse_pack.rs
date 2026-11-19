//! Host packing for fixed_csr_spmv. Input inverse is row-major in free-body order.
#[derive(Debug)]
pub struct Packed { pub rows:u32, pub index:Vec<u32>, pub offset:Vec<u32>, pub value:Vec<f32>, pub transpose_pair:Vec<u32>, pub transpose_offset:Vec<u32> }

pub fn pack(a:&[f64], free:&[usize], roots:&[usize], nbody:usize)->Result<Packed,&'static str> {
 let n=free.len().checked_mul(6).ok_or("shape")?;
 if a.len()!=n.checked_mul(n).ok_or("shape")? {return Err("shape")}
 let rows=nbody.checked_mul(2).ok_or("shape")?;
 if rows>u32::MAX as usize || rows.checked_mul(rows).ok_or("shape")?>u32::MAX as usize {return Err("shape")}
 let mut map=vec![None;nbody];
 for (i,&b) in free.iter().enumerate(){if b>=nbody || map[b].is_some(){return Err("mapping")} map[b]=Some(i);}
 let mut root=vec![false;nbody];
 for &b in roots {if b>=nbody || root[b] || map[b].is_some(){return Err("mapping")} root[b]=true;}
 if (0..nbody).any(|b|map[b].is_none()&&!root[b]) {return Err("mapping")}
 let mut sym=vec![0f32;n*n];
 for i in 0..n {for j in 0..=i {let x=a[i*n+j]; let y=a[j*n+i]; if !x.is_finite()||!y.is_finite(){return Err("nonfinite")} let v=(0.5*x+0.5*y) as f32; if !v.is_finite(){return Err("nonfinite")} sym[i*n+j]=v;sym[j*n+i]=v;}}
 let mut l=vec![0f64;n*n];
 for i in 0..n {for j in 0..=i {let mut v=sym[i*n+j] as f64; for k in 0..j {v-=l[i*n+k]*l[j*n+k];} if i==j {if !v.is_finite()||v<=0.0{return Err("not positive definite")} l[i*n+j]=v.sqrt();} else {l[i*n+j]=v/l[j*n+j]; if !l[i*n+j].is_finite(){return Err("nonfinite")}}}}
 let mut p=Packed{rows:rows as u32,index:vec![],offset:vec![0],value:vec![],transpose_pair:vec![],transpose_offset:vec![0]};
 let mut trans=vec![Vec::<(u32,u32)>::new();rows];
 for r in 0..rows {for c in r..rows {if let (Some(fr),Some(fc))=(map[r/2],map[c/2]) {let slot=p.index.len() as u32;p.index.push(c as u32);for j in 0..3 {for i in 0..3 {p.value.push(sym[(fr*6+(r%2)*3+i)*n+fc*6+(c%2)*3+j]);}} if r!=c {trans[c].push((r as u32,slot));}}}p.offset.push(p.index.len() as u32);}
 for entries in trans {for (r,s) in entries {p.transpose_pair.extend_from_slice(&[r,s]);}p.transpose_offset.push((p.transpose_pair.len()/2) as u32);}
 Ok(p)
}
#[cfg(test)] mod tests {
 use super::*;
 fn apply(p:&Packed,x:&[f32])->Vec<f32>{let mut y=vec![0.;x.len()];for r in 0..p.rows as usize {for k in p.offset[r] as usize..p.offset[r+1] as usize {let c=p.index[k] as usize;for i in 0..3 {for j in 0..3 {y[3*r+i]+=p.value[9*k+3*j+i]*x[3*c+j];}}}for t in p.transpose_offset[r] as usize..p.transpose_offset[r+1] as usize {let c=p.transpose_pair[2*t] as usize;let k=p.transpose_pair[2*t+1] as usize;for i in 0..3 {for j in 0..3 {y[3*r+i]+=p.value[9*k+3*i+j]*x[3*c+j];}}}}y}
 #[test] fn dense_oracle(){let n=12;let mut a=vec![0.;n*n];for i in 0..n {for j in 0..n {a[i*n+j]=if i==j {20.} else {((i.min(j)*n+i.max(j)+1) as f64)*0.001};}}let p=pack(&a,&[2,0],&[1],3).unwrap();let x:Vec<f32>=(0..18).map(|i|i as f32-5.).collect();let y=apply(&p,&x);for b in 0..3 {for i in 0..6 {let mut expected=0.;if b!=1 {let fi=if b==2 {0}else{1};for fj in 0..2 {let bj=if fj==0 {2}else{0};for j in 0..6 {expected+=(a[(fi*6+i)*n+fj*6+j] as f32)*x[bj*6+j];}}}assert!((y[b*6+i]-expected).abs()<0.001);}}}
 #[test] fn rejects(){assert!(pack(&[1.],&[0],&[],1).is_err());let mut a=vec![0.;36];for i in 0..6 {a[i*6+i]=1.;}a[0]=f64::NAN;assert!(pack(&a,&[0],&[],1).is_err());a[0]=1e-50;assert!(pack(&a,&[0],&[],1).is_err());assert!(pack(&a,&[0],&[0],1).is_err());}
}

