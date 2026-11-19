//! Packed existing host factor; no factorization or device operations.
use super::body_tree_factor::{TreeView, Unsupported};

#[derive(Debug)]
pub struct PackedFactor {
 pub version:u32, pub body_count:u32, pub node_count:u32, pub component_count:u32,
 pub body_index:Vec<u32>, pub component_offsets:Vec<u32>, pub parent:Vec<u32>,
 pub depth:Vec<u32>, pub max_depth:Vec<u32>, pub child_offsets:Vec<u32>, pub children:Vec<u32>,
 pub root_body_indices:Vec<u32>, pub l_words:Vec<u32>, pub w_words:Vec<u32>,
}
fn err(s:&str)->Unsupported {Unsupported(s.into())}
fn word(x:usize)->Result<u32,Unsupported>{u32::try_from(x).map_err(|_|err("extent overflow"))}
impl PackedFactor {
 pub fn from_view(v:TreeView<'_>,free:&[usize],bodies:usize)->Result<Self,Unsupported>{
  let n=free.len();if n==0 || n.checked_mul(72).filter(|&x|u32::try_from(x).is_ok()).is_none(){return Err(err("free extent refused"));}
  let body_count=word(bodies)?;let node_count=word(n)?;
  if v.parent.len()!=n||v.order.len()!=n||v.l.len()!=n||v.w.len()!=n{return Err(err("factor extent"));}
  let mut used=vec![false;bodies];for &b in free {if b>=bodies||used[b]{return Err(err("free body mapping"));}used[b]=true;}
  let root_body_indices=(0..bodies).filter(|&b|!used[b]).map(word).collect::<Result<Vec<_>,_>>()?;
  let mut inverse=vec![usize::MAX;n];for (i,&u) in v.order.iter().enumerate(){if u>=n||inverse[u]!=usize::MAX{return Err(err("factor order"));}inverse[u]=i;}
  let mut body_index=Vec::with_capacity(n);let mut parent=Vec::with_capacity(n);let mut depth=Vec::with_capacity(n);
  let mut component_offsets=Vec::new();let mut max_depth:Vec<u32>=Vec::new();let mut l_words=Vec::with_capacity(n*72);let mut w_words=Vec::with_capacity(n*72);
  let mut component=vec![usize::MAX;n];
  for (i,&u) in v.order.iter().enumerate(){
   body_index.push(word(free[u])?);
   let d=match v.parent[u] {None=>{component_offsets.push(word(i)?);max_depth.push(0);component[u]=max_depth.len()-1;parent.push(u32::MAX);0},Some(p)=>{if p>=n||inverse[p]>=i||component[p]==usize::MAX{return Err(err("invalid parent"));}let c=max_depth.len().checked_sub(1).ok_or_else(||err("invalid component"))?;if component[p]!=c{return Err(err("parent component"));}component[u]=c;parent.push(word(inverse[p])?);depth[inverse[p]]+1}};
   depth.push(d);let c=component[u];max_depth[c]=max_depth[c].max(d);
   for (block,out) in [(&v.l[u],&mut l_words),(&v.w[u],&mut w_words)] {for row in block {for &x in row {if !x.is_finite(){return Err(err("nonfinite factor"));}let bits=x.to_bits();out.push(bits as u32);out.push((bits>>32) as u32);}}}
   if (0..6).any(|k|v.l[u][k][k]<=0.){return Err(err("nonpositive pivot"));}
  }
  component_offsets.push(word(n)?);if component_offsets.windows(2).any(|w|w[1]-w[0]>170){return Err(err("component extent"));}let component_count=word(max_depth.len())?;
  let mut by_parent=vec![Vec::new();n];for i in (0..n).rev(){if parent[i]!=u32::MAX{by_parent[parent[i] as usize].push(word(i)?);}}
  let mut child_offsets=vec![0];let mut children=Vec::new();for child in by_parent {children.extend(child);child_offsets.push(word(children.len())?);}
  Ok(Self{version:1,body_count,node_count,component_count,body_index,component_offsets,parent,depth,max_depth,child_offsets,children,root_body_indices,l_words,w_words})
 }
}
#[cfg(test)] mod tests {
 use super::*;use super::super::body_tree_factor::{Tree,Pair,Block};
 fn diag(x:f64)->Block {let mut a=[[0.;6];6];for i in 0..6{a[i][i]=x+i as f64*0.1;}a}
 fn block(w:&[u32],n:usize,i:usize,j:usize)->f64{let k=72*n+2*(6*i+j);f64::from_bits((w[k] as u64)|((w[k+1] as u64)<<32))}
 fn apply(p:&PackedFactor,r:&[f64])->Vec<f64>{let n=p.node_count as usize;let mut q=vec![[0.;6];n];let mut x=q.clone();for i in 0..n{q[i].copy_from_slice(&r[6*p.body_index[i] as usize..6*p.body_index[i] as usize+6]);}
  for u in (0..n).rev(){let pa=p.parent[u];if pa!=u32::MAX{for j in 0..6{let mut t=0.;for i in 0..6{t+=block(&p.w_words,u,i,j)*q[u][i];}q[pa as usize][j]-=t;}}}
  for u in 0..n{let mut y=q[u];for i in 0..6{for j in 0..i{y[i]-=block(&p.l_words,u,i,j)*y[j];}y[i]/=block(&p.l_words,u,i,i);}for i in (0..6).rev(){for j in i+1..6{y[i]-=block(&p.l_words,u,j,i)*y[j];}y[i]/=block(&p.l_words,u,i,i);}if p.parent[u]!=u32::MAX{for i in 0..6{for j in 0..6{y[i]-=block(&p.w_words,u,i,j)*x[p.parent[u] as usize][j];}}}x[u]=y;}
  let mut out=vec![0.;r.len()];for u in 0..n{out[6*p.body_index[u] as usize..6*p.body_index[u] as usize+6].copy_from_slice(&x[u]);}out }
 #[test] fn packed_matches_existing_forest_and_roots(){let d=[diag(4.),diag(5.),diag(6.),diag(7.)];let pairs=[Pair{a:0,b:1,ab:diag(-0.2)},Pair{a:1,b:2,ab:diag(-0.3)}];let t=Tree::factor(&d,&pairs).unwrap();let free=[4,0,3,2];let p=PackedFactor::from_view(t.view(),&free,6).unwrap();assert_eq!(p.component_count,2);assert_eq!(p.root_body_indices,vec![1,5]);let r:Vec<f64>=(0..36).map(|i|i as f64*0.25-2.).collect();let packed=apply(&p,&r);let reduced:Vec<_>=free.iter().flat_map(|&b|r[6*b..6*b+6].iter().copied()).collect();let expected=t.apply(&reduced).unwrap();for (i,&b) in free.iter().enumerate(){assert_eq!(&packed[6*b..6*b+6],&expected[6*i..6*i+6]);}assert!(packed[6..12].iter().all(|&x|x==0.));for parent in 0..p.node_count as usize{let children=&p.children[p.child_offsets[parent] as usize..p.child_offsets[parent+1] as usize];assert!(children.windows(2).all(|x|x[0]>x[1]));}}
 #[test] fn accepts_more_than_170_nodes_in_small_components_and_rejects_one_large_component(){
  let n=171usize;let d=vec![diag(4.);n];let free:Vec<_>=(0..n).collect();
  let forest=Tree::factor(&d,&[]).unwrap();let packed=PackedFactor::from_view(forest.view(),&free,n).unwrap();
  assert_eq!(packed.node_count,171);assert_eq!(packed.component_count,171);
  let pairs:Vec<_>=(0..n-1).map(|i|Pair{a:i,b:i+1,ab:diag(-0.01)}).collect();
  let chain=Tree::factor(&d,&pairs).unwrap();assert!(PackedFactor::from_view(chain.view(),&free,n).is_err());
 }
 #[test] fn rejects_invalid_mapping_and_extent(){let t=Tree::factor(&[diag(1.),diag(2.)],&[]).unwrap();assert!(PackedFactor::from_view(t.view(),&[0,0],2).is_err());assert!(PackedFactor::from_view(t.view(),&[0,2],2).is_err());assert!(PackedFactor::from_view(t.view(),&[],2).is_err());}
 #[test] fn rejects_invalid_factor_data(){
  let l=[diag(1.),diag(2.)];let w=[diag(0.),diag(0.)];let parent=[None,Some(0)];let order=[0,1];
  let view=||TreeView{parent:&parent,order:&order,l:&l,w:&w};
  assert!(PackedFactor::from_view(view(),&[0,1],2).is_ok());
  let invalid=[Some(1),None];assert!(PackedFactor::from_view(TreeView{parent:&invalid,order:&order,l:&l,w:&w},&[0,1],2).is_err());
  let mut bad=l;bad[1][2][2]=f64::NAN;assert!(PackedFactor::from_view(TreeView{parent:&parent,order:&order,l:&bad,w:&w},&[0,1],2).is_err());
  bad=l;bad[0][0][0]=0.;assert!(PackedFactor::from_view(TreeView{parent:&parent,order:&order,l:&bad,w:&w},&[0,1],2).is_err());
  let free:Vec<_>=(0..171).collect();assert!(PackedFactor::from_view(view(),&free,171).is_err());
 }

}
