//! Sparse raw directed projection of frozen full-space and joint blocks.
//! Block values are column-major 3x3, duplicates deliberately retained.
#[derive(Clone, Copy, Debug)]
pub struct DirectedBlock {pub row:u32,pub col:u32,pub value:[f32;9]}
#[derive(Clone, Copy)]
pub struct FullBlock {pub row:usize,pub col:usize,pub value:[f32;9]}
fn basis(v:usize,vbody:&[u32],cloth:&[u32],prot:&[f32],base:usize,bodies:usize)->Result<Vec<(u32,[[f64;3];3])>, &'static str>{
 if v>=vbody.len()||cloth.len()!=vbody.len()||prot.len()!=3*vbody.len(){return Err("map extent")}
 let b=vbody[v];let id=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
 if b==u32::MAX {let x=cloth[v] as usize;if x%3!=0||x.checked_add(3).map_or(true,|e|e>base){return Err("cloth offset")}return Ok(vec![((x/3) as u32,id)])}
 let b=b as usize;if b>=bodies{return Err("body index")};let p=&prot[3*v..3*v+3];if p.iter().any(|x|!x.is_finite()){return Err("prot nonfinite")};
 let(x,y,z)=(p[0] as f64,p[1] as f64,p[2] as f64);let rot=[[0.,z,-y],[-z,0.,x],[y,-x,0.]];
 let row=base.checked_div(3).and_then(|n|n.checked_add(2*b)).ok_or("row overflow")?;if row+1>u32::MAX as usize{return Err("row overflow")};Ok(vec![(row as u32,id),((row+1) as u32,rot)])
}
pub fn project_block(vbody:&[u32],cloth:&[u32],prot:&[f32],base:usize,bodies:usize,b:FullBlock,out:&mut Vec<DirectedBlock>)->Result<(), &'static str>{
 if base%3!=0||b.value.iter().any(|x|!x.is_finite()){return Err("invalid input")}
 let left=basis(b.row,vbody,cloth,prot,base,bodies)?;let right=basis(b.col,vbody,cloth,prot,base,bodies)?;
 for (ri,jr) in &left {for (ci,jc) in &right {let mut v=[0f32;9];for i in 0..3{for j in 0..3{let mut x=0f64;for r in 0..3{for c in 0..3{x+=jr[r][i]*b.value[3*c+r] as f64*jc[c][j];}}v[3*j+i]=x as f32;if !v[3*j+i].is_finite(){return Err("projection nonfinite")}}}out.push(DirectedBlock{row:*ri,col:*ci,value:v});}}
 Ok(())
}
pub fn project_joint(ids:&[u32],h:&[f32],dim:usize,out:&mut Vec<DirectedBlock>)->Result<(), &'static str>{
 let n=ids.len();if n>24||h.len()!=n*n||dim%3!=0{return Err("joint extent")};for i in 0..n{for j in 0..n{let x=h[n*i+j];if !x.is_finite(){return Err("joint nonfinite")};let(a,b)=(ids[i] as usize,ids[j] as usize);if a==u32::MAX as usize||b==u32::MAX as usize{continue}if a>=dim||b>=dim{return Err("joint id")};let mut v=[0f32;9];v[3*(b%3)+a%3]=x;out.push(DirectedBlock{row:(a/3) as u32,col:(b/3) as u32,value:v});}}Ok(())
}
/// Construct masked E^T F^T diag(w) F E as directed full 3x3 blocks.
pub fn tet_blocks(vertices:[usize;4],mask:&[u32],factor:&[f32],weight:&[f32])->Result<Vec<FullBlock>, &'static str>{
 if factor.len()!=81||weight.len()!=9||vertices.iter().any(|&v|v>=mask.len())||factor.iter().any(|v|!v.is_finite())||weight.iter().any(|v|!v.is_finite()||*v<0.){return Err("tet input")}
 let mut h=[[0f64;9];9];for i in 0..9{for j in 0..9{for k in 0..9{h[i][j]+=factor[9*k+i] as f64*weight[k] as f64*factor[9*k+j] as f64;}}}
 let mut out=Vec::new();for a in 0..4{if mask[vertices[a]]!=0{continue}for b in 0..4{if mask[vertices[b]]!=0{continue}let mut value=[0f32;9];for i in 0..3{for j in 0..3{let mut x=0.;for u in 0..3{let ea=if a==0{-1.}else if a==u+1{1.}else{0.};for v in 0..3{let eb=if b==0{-1.}else if b==v+1{1.}else{0.};x+=ea*h[3*u+i][3*v+j]*eb;}}value[3*j+i]=x as f32;if !value[3*j+i].is_finite(){return Err("tet nonfinite")}}}out.push(FullBlock{row:vertices[a],col:vertices[b],value});}}Ok(out)
}
#[cfg(test)] mod tests {use super::*;
 #[test] fn mixed_directed_projection_and_duplicate_ids(){let vb=[u32::MAX,0];let cloth=[0,0];let p=[0.,0.,0.,1.,2.,3.];let mut out=Vec::new();project_block(&vb,&cloth,&p,3,1,FullBlock{row:0,col:1,value:[1.,2.,3.,4.,5.,6.,7.,8.,9.]},&mut out).unwrap();assert_eq!(out.len(),2);assert_eq!((out[0].row,out[0].col),(0,1));assert_eq!(out[0].value,[1.,2.,3.,4.,5.,6.,7.,8.,9.]);assert_eq!((out[1].row,out[1].col),(0,2));let mut j=Vec::new();project_joint(&[0,3,3],&[1.;9],9,&mut j).unwrap();assert_eq!(j.iter().filter(|x|x.row==1&&x.col==1).count(),4);}
 #[test] fn tet_mask_and_energy(){let mut f=[0f32;81];for i in 0..9{f[9*i+i]=1.;}let blocks=tet_blocks([0,1,2,3],&[0,0,1,0],&f,&[1.;9]).unwrap();assert_eq!(blocks.len(),9);assert!(blocks.iter().all(|b|b.row!=2&&b.col!=2));let x=[[1.,2.,3.],[4.,5.,6.],[9.,9.,9.],[-1.,0.,1.]];let mut y=[[0f64;3];4];for b in &blocks{for i in 0..3{for j in 0..3{y[b.row][i]+=b.value[3*j+i] as f64*x[b.col][j];}}}for i in 0..3{let v0=x[0][i];assert!((y[0][i]-(3.*v0-x[1][i]-x[3][i])).abs()<1e-6);assert!((y[1][i]-(x[1][i]-v0)).abs()<1e-6);assert!((y[3][i]-(x[3][i]-v0)).abs()<1e-6);}}
 #[test] fn tet_nondiagonal_mixed_matches_evaluate_gather(){
  let vertices=[0,1,2,3];let masks=[0,0,1,0];let vb=[u32::MAX,0,u32::MAX,1];let cloth=[0,0,3,0];let prot=[0.,0.,0., 1.,2.,-1., 0.,0.,0., -2.,0.5,1.];let base=6;let bodies=2;
  let mut factor=[0f32;81];for i in 0..9{for j in 0..9{factor[9*i+j]=if i==j{1.+i as f32*0.03}else{((i*11+j*7)%13) as f32*0.007-0.04};}}let mut weights=[0f32;9];for i in 0..9{weights[i]=0.3+i as f32*0.11;}
  let blocks=tet_blocks(vertices,&masks,&factor,&weights).unwrap();let mut projected=Vec::new();for b in blocks{project_block(&vb,&cloth,&prot,base,bodies,b,&mut projected).unwrap();}
  let xr:Vec<f64>=(0..18).map(|i|(i as f64-8.)*0.04).collect();let mut actual=vec![0f64;18];for b in projected{for i in 0..3{for j in 0..3{actual[3*b.row as usize+i]+=b.value[3*j+i] as f64*xr[3*b.col as usize+j];}}}
  let mut full=[[0f64;3];4];full[0].copy_from_slice(&xr[..3]);full[2].copy_from_slice(&xr[3..6]);for (v,b) in [(1usize,0usize),(3,1)]{let off=6+6*b;let p=&prot[3*v..3*v+3];let(w1,w2,w3)=(xr[off+3],xr[off+4],xr[off+5]);full[v]=[xr[off]+p[2] as f64*w2-p[1] as f64*w3,xr[off+1]-p[2] as f64*w1+p[0] as f64*w3,xr[off+2]+p[1] as f64*w1-p[0] as f64*w2];}
  full[2]=[0.;3];let mut delta=[0f64;9];for v in 1..4{for d in 0..3{delta[3*(v-1)+d]=full[v][d]-full[0][d];}}let mut response=[0f64;9];for k in 0..9{let dot=(0..9).map(|j|factor[9*k+j] as f64*delta[j]).sum::<f64>();for j in 0..9{response[j]+=factor[9*k+j] as f64*weights[k] as f64*dot;}}
  let mut force=[[0f64;3];4];for d in 0..3{force[0][d]=-(response[d]+response[3+d]+response[6+d]);force[1][d]=response[d];force[3][d]=response[6+d];}let mut expected=vec![0f64;18];expected[..3].copy_from_slice(&force[0]);for (v,b) in [(1usize,0usize),(3,1)]{let off=6+6*b;let p=&prot[3*v..3*v+3];let q=force[v];expected[off..off+3].copy_from_slice(&q);expected[off+3]=-p[2] as f64*q[1]+p[1] as f64*q[2];expected[off+4]=p[2] as f64*q[0]-p[0] as f64*q[2];expected[off+5]=-p[1] as f64*q[0]+p[0] as f64*q[1];}
  for i in 0..18{assert!((actual[i]-expected[i]).abs()<2e-6,"{i}: {} != {}",actual[i],expected[i]);}
 }

}
