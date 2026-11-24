//! Host-only shadow prototype; this does not mutate the live solver.
use super::body_tree_factor;
use body_tree_factor::{Block, Pair, Tree, Unsupported};
use std::collections::BTreeMap;

#[derive(serde::Serialize,serde::Deserialize,PartialEq,Debug)]
pub struct ShadowDomain {version:u32,scene_uuid:String,scene_seal:String,nrow:usize,ncloth:usize,bodies:usize,dim:usize,body_base:usize,vbody:Vec<u32>,cloth_off:Vec<u32>,free:Vec<usize>,projection:bool,lock:bool,hard:bool}
#[derive(serde::Serialize,serde::Deserialize)]
struct LaggedRecord {domain:ShadowDomain,columns:Vec<f32>,source_solve:usize,start_time:f64,sample_time:f32}
pub fn domain(map:&super::rigid_map::RigidMap,j:super::joints::View,free:Vec<usize>)->Result<ShadowDomain,String>{
    let (u,s)=j.shadow_identity.ok_or("lagged shadow requires sealed identity")?;
    let scene_uuid=String::from_utf8(u.to_vec()).map_err(|_|"scene identity invalid")?;
    let scene_seal=String::from_utf8(s.to_vec()).map_err(|_|"sealed identity invalid")?;
    if !scene_seal.bytes().all(|b|b.is_ascii_hexdigit()) {return Err("sealed identity invalid".into());}
    Ok(ShadowDomain{version:1,scene_uuid,scene_seal,nrow:map.nrow,ncloth:map.n_cloth,bodies:map.n_bodies,dim:map.dim,body_base:map.body_base,vbody:map.vbody.clone(),cloth_off:map.cloth_off.clone(),free,projection:map.needs_projection(),lock:false,hard:false})
}
pub fn save_lagged(path:&std::path::Path,domain:ShadowDomain,columns:Vec<f32>,solve:usize,j:super::joints::View)->Result<(),String>{
    use std::io::Write;
    let r=LaggedRecord{domain,columns,source_solve:solve,start_time:j.start_time_s,sample_time:j.sample_time_s};
    let mut f=std::fs::OpenOptions::new().write(true).create_new(true).open(path).map_err(|e|e.to_string())?;
    serde_json::to_writer(&mut f,&r).map_err(|e|e.to_string())?;f.flush().map_err(|e|e.to_string())
}
pub fn load_lagged(path:&std::path::Path,expected:&ShadowDomain)->Result<(Vec<f32>,usize,f64,f32),String>{
    let meta=std::fs::metadata(path).map_err(|e|e.to_string())?;if meta.len()>32*1024*1024{return Err("lagged record exceeds 32MiB".into());}
    let f=std::fs::File::open(path).map_err(|e|e.to_string())?;let r:LaggedRecord=serde_json::from_reader(f).map_err(|e|e.to_string())?;
    if &r.domain!=expected{return Err("lagged domain mismatch".into());}if !r.start_time.is_finite()||!r.sample_time.is_finite(){return Err("lagged time invalid".into());}
    Ok((r.columns,r.source_solve,r.start_time,r.sample_time))
}

pub fn fault_snapshot<D:ppf_cts_compute::Device>(device:&mut D,op:&super::operator::Operator,j:super::joints::View)->Result<Vec<Vec<u32>>,String>{
    let mut handles=j.shadow_fault_handles();
    if let Some(tet)=op.tet_factor {handles.push(tet.capture_snapshot().fault);}
    let mut all=Vec::with_capacity(handles.len());
    for h in handles {let n=h.size as usize;if n>10_000_000{return Err("fault extent refused".into());}let mut values=vec![0u32;n];
        if n>0 {let bytes=unsafe{std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(),n*4)};device.read(h,0,bytes).map_err(|e|e.to_string())?;}
        all.push(values);
    } Ok(all)
}

/// One-shot diagnostic capture of the *already assembled* BJ body blocks and
/// the ordinary-ball factors at the same Newton pose. No model values change.
pub unsafe fn capture<D:ppf_cts_compute::Device>(device:&mut D, blocks:ppf_cts_compute::Handle,
    bodies:usize, roots:Option<super::roots::View>, j:super::joints::View)->Result<FreeForest,String>{
    use ppf_cts_compute::{AllocLabel,Buffer};
    use crate::joint_hard_coefficients::SmoothFactors;
    let mut raw=vec![0f32;bodies.checked_mul(36).ok_or("block extent overflow")?];
    let bytes=unsafe{std::slice::from_raw_parts_mut(raw.as_mut_ptr().cast::<u8>(),raw.len()*4)};
    device.read(blocks,0,bytes).map_err(|e|e.to_string())?;
    let mut diag=vec![[[0f64;6];6];bodies];
    for b in 0..bodies {for i in 0..6 {for k in 0..6 {
        diag[b][i][k]=0.5*((raw[36*b+6*i+k] as f64)+(raw[36*b+6*k+i] as f64));
    }}}
    let mut fixed=vec![false;bodies];
    if let Some(root)=roots {let mut rs=vec![crate::joint_root_motion_gpu::RootTargetGpu::default();root.count as usize];
        let bytes=unsafe{std::slice::from_raw_parts_mut(rs.as_mut_ptr().cast::<u8>(),rs.len()*std::mem::size_of::<crate::joint_root_motion_gpu::RootTargetGpu>())};
        device.read(root.targets,0,bytes).map_err(|e|e.to_string())?;
        for r in rs {if r.enabled!=0 {let b=r.body as usize;if b>=bodies{return Err("root body out of range".into())} fixed[b]=true;}}
    }
    let mut out=Buffer::<SmoothFactors>::default();out.size(device,j.count as usize,AllocLabel("shadow.tree.ball_factors")).map_err(|e|e.to_string())?;
    let args=super::kernels::HardFrozenSmoothFactorsArgs {ball:j.ball,angular:j.angular,control:j.control,twist_context:j.twist_context,state:j.state,reference:j.reference,bodies:j.bodies,dt:j.dt,control_count:0,output:out.handle(),count:j.count,seam_arena_count:0};
    let mut fs=vec![SmoothFactors::default();j.count as usize];
    let captured=(|| -> Result<(),String> {if j.count>0 {device.launch("shadow.tree.ball_factors",&args,j.count).map_err(|e|e.to_string())?;}
        let bytes=unsafe{std::slice::from_raw_parts_mut(fs.as_mut_ptr().cast::<u8>(),fs.len()*std::mem::size_of::<SmoothFactors>())};
        if !bytes.is_empty(){device.read(out.handle(),0,bytes).map_err(|e|e.to_string())?;}Ok(())})();
    let freed=out.free(device).map_err(|e|e.to_string());captured?;freed?;
    fn lift(p:[f32;3])->[[f64;6];3] {let (x,y,z)=(p[0] as f64,p[1] as f64,p[2] as f64);[[1.,0.,0.,0.,z,-y],[0.,1.,0.,-z,0.,x],[0.,0.,1.,y,-x,0.]]}
    let mut edges=Vec::new();for f in fs {if f.fault!=0{return Err(format!("ball factor fault {}",f.fault));}if f.ball_active==0{continue;}if !f.ball_k.is_finite()||f.ball_k<=0.{return Err("invalid ball stiffness".into());}
        let a=lift(f.p_a);let b=lift(f.p_b);let mut ab=[[0f64;6];6];for i in 0..6{for k in 0..6{for d in 0..3{ab[i][k]-=a[d][i]*(f.ball_k as f64)*b[d][k];}}}
        edges.push(Pair{a:f.body_a as usize,b:f.body_b as usize,ab});}
    prepare(&diag,&fixed,&edges).map_err(|e|e.0)
}

pub struct FreeForest { tree: Tree, free: Vec<usize>, bodies: usize, selected_edges: Vec<(usize,usize)>, pub omitted_cycles: usize }

pub struct DenseBody {free:Vec<usize>, bodies:usize, l:Vec<f64>}
impl DenseBody {
    /// Raw column-probed QAQ body principal, f64 symmetric M only; no jitter.
    pub fn factor(free:Vec<usize>,bodies:usize,columns:&[f32])->Result<(Self,f64,f64),String>{
        let n=free.len();if n==0||n>1024||columns.len()!=n.checked_mul(n).ok_or("dense extent overflow")? {return Err("dense extent refused".into());}
        let slots=bodies.checked_mul(6).ok_or("body overflow")?;let mut seen=vec![false;slots];
        for &i in &free {if i>=slots || seen[i] {return Err("invalid free index".into());}seen[i]=true;}
        if columns.iter().any(|v|!v.is_finite()){return Err("dense input invalid".into());}
        let(mut skew,mut norm)=(0f64,0f64);let mut l=vec![0f64;n*n];
        for i in 0..n{for j in 0..n{let a=columns[j*n+i] as f64;let b=columns[i*n+j] as f64;skew=skew.max((a-b).abs());norm=norm.max(a.abs());}}
        for i in 0..n{for j in 0..=i{let mut v=0.5*((columns[j*n+i] as f64)+(columns[i*n+j] as f64));for k in 0..j{v-=l[i*n+k]*l[j*n+k];}if i==j{if !v.is_finite()||v<=0.{return Err(format!("dense nonpositive Schur pivot {i}"));}l[i*n+j]=v.sqrt();}else{l[i*n+j]=v/l[j*n+j];if !l[i*n+j].is_finite(){return Err("dense factor nonfinite".into());}}}}
        Ok((Self{free,bodies,l},skew,norm))
    }
    pub fn apply_mixed(&self,r:&[f32],bj:&[f32],base:usize)->Result<Vec<f32>,String>{
        let n=self.free.len();let end=base.checked_add(self.bodies.checked_mul(6).ok_or("body overflow")?).ok_or("extent overflow")?;
        if r.len()!=end||bj.len()!=end{return Err("mixed extent mismatch".into());}
        let mut y:Vec<f64>=self.free.iter().map(|&i|r[base+i] as f64).collect();
        for i in 0..n{for j in 0..i{y[i]-=self.l[i*n+j]*y[j];}y[i]/=self.l[i*n+i];}
        for i in (0..n).rev(){for j in i+1..n{y[i]-=self.l[j*n+i]*y[j];}y[i]/=self.l[i*n+i];}
        let mut out=bj.to_vec();out[base..].fill(0.);for (i,&slot) in self.free.iter().enumerate(){out[base+slot]=y[i] as f32;if !out[base+slot].is_finite(){return Err("dense result nonfinite".into());}}Ok(out)
    }
}

/// Diagonals must be the already assembled BJ self-blocks, not joint additions.
/// Root lanes are eliminated; edges incident to roots contribute no crossblock.
pub fn prepare(diag: &[Block], fixed: &[bool], pairs: &[Pair]) -> Result<FreeForest, Unsupported> {
    let n=diag.len();
    if fixed.len()!=n { return Err(Unsupported("root mask length".into())); }
    let free:Vec<_>=(0..n).filter(|&i|!fixed[i]).collect();
    if free.is_empty() { return Err(Unsupported("no free bodies".into())); }
    let mut map=vec![usize::MAX;n];
    for (i,&b) in free.iter().enumerate(){map[b]=i;}
    let mut edges=BTreeMap::<(usize,usize),Block>::new();
    for p in pairs {
        if p.a>=n || p.b>=n || p.a==p.b || p.ab.iter().flatten().any(|x|!x.is_finite()) {return Err(Unsupported("invalid edge".into()));}
        if fixed[p.a] || fixed[p.b] {continue;}
        let (a,b,rev)=if map[p.a]<map[p.b] {(map[p.a],map[p.b],false)} else {(map[p.b],map[p.a],true)};
        let block=edges.entry((a,b)).or_insert([[0.;6];6]);
        for i in 0..6 {for j in 0..6 {block[i][j]+=if rev {p.ab[j][i]} else {p.ab[i][j]};}}
        if block.iter().flatten().any(|x|!x.is_finite()) {return Err(Unsupported("edge accumulation overflow".into()));}
    }
    let mut parent:Vec<_>=(0..free.len()).collect();
    fn find(p:&mut [usize],mut x:usize)->usize {while p[x]!=x {p[x]=p[p[x]];x=p[x];} x}
    let mut forest=Vec::new(); let mut omitted=0;
    for ((a,b),ab) in edges {let x=find(&mut parent,a);let y=find(&mut parent,b);if x==y {omitted+=1;} else {parent[x]=y;forest.push(Pair{a,b,ab});}}
    let d:Vec<_>=free.iter().map(|&i|diag[i]).collect();
    let selected_edges=forest.iter().map(|p|(p.a,p.b)).collect();
    Ok(FreeForest {tree:Tree::factor(&d,&forest)?,free,bodies:n,selected_edges,omitted_cycles:omitted})
}
impl FreeForest {
    pub fn factor_view(&self) -> (body_tree_factor::TreeView<'_>, &[usize], usize) { (self.tree.view(), &self.free, self.bodies) }
    /// Build a capacity-bounded forest from current directed projected blocks.
    /// Capacity cuts affect only the preconditioner; the original operator is untouched.
    pub fn from_sparse_weighted(matrix:&super::body_block_projection::SparseBodyMatrix, component_capacity:usize)->Result<Self,Unsupported>{
        let n=matrix.free.len();
        if n==0 || component_capacity==0 {return Err(Unsupported("empty sparse domain or capacity".into()));}
        let mut seen=std::collections::BTreeSet::new();
        for &b in &matrix.free {if b>=matrix.body_count || !seen.insert(b){return Err(Unsupported("invalid free body mapping".into()));}}
        for (&(a,b),block) in &matrix.blocks {if a>=n||b>=n||block.iter().flatten().any(|v|!v.is_finite()){return Err(Unsupported("invalid sparse block".into()));}}
        let zero=[[0f32;6];6];let mut diag=vec![[[0f64;6];6];n];let mut ls=diag.clone();
        for a in 0..n {let d=matrix.blocks.get(&(a,a)).unwrap_or(&zero);
            for i in 0..6{for j in 0..6{diag[a][i][j]=0.5*((d[i][j] as f64)+(d[j][i] as f64));}}
            for i in 0..6{for j in 0..=i{let mut v=diag[a][i][j];for k in 0..j{v-=ls[a][i][k]*ls[a][j][k];}if i==j{if !v.is_finite()||v<=0.{return Err(Unsupported("nonpositive diagonal pivot".into()));}ls[a][i][j]=v.sqrt();}else{ls[a][i][j]=v/ls[a][j][j];}}}
        }
        let mut edges=std::collections::BTreeSet::new();
        for &(a,b) in matrix.blocks.keys(){if a!=b{edges.insert((a.min(b),a.max(b)));}}
        let mut candidates=Vec::new();
        candidates.try_reserve_exact(edges.len()).map_err(|_| Unsupported("forest candidate allocation".into()))?;
        for (a,b) in edges {let ab=matrix.blocks.get(&(a,b)).unwrap_or(&zero);let ba=matrix.blocks.get(&(b,a)).unwrap_or(&zero);let mut c=[[0f64;6];6];
            for i in 0..6{for j in 0..6{c[i][j]=0.5*((ab[i][j] as f64)+(ba[j][i] as f64));}}
            if c.iter().flatten().all(|v|*v==0.){continue;}
            let mut x=c;for col in 0..6{for i in 0..6{for k in 0..i{x[i][col]-=ls[a][i][k]*x[k][col];}x[i][col]/=ls[a][i][i];}}
            let mut y=x;for row in 0..6{for i in 0..6{for k in 0..i{y[row][i]-=ls[b][i][k]*y[row][k];}y[row][i]/=ls[b][i][i];}}
            let w=y.iter().flatten().map(|v|v*v).sum::<f64>().sqrt();if !w.is_finite(){return Err(Unsupported("nonfinite edge weight".into()));}candidates.push((w,a,b,c));
        }
        candidates.sort_by(|x,y|y.0.total_cmp(&x.0).then_with(||matrix.free[x.1].cmp(&matrix.free[y.1])).then_with(||matrix.free[x.2].cmp(&matrix.free[y.2])));
        let mut parents:Vec<_>=(0..n).collect();let mut sizes=vec![1usize;n];
        fn find(p:&mut [usize],mut x:usize)->usize{while p[x]!=x{p[x]=p[p[x]];x=p[x];}x}
        let mut pairs=Vec::new();let mut skipped=0;
        for (_,a,b,ab) in candidates {let x=find(&mut parents,a);let y=find(&mut parents,b);if x==y||sizes[x].checked_add(sizes[y]).map_or(true,|v|v>component_capacity){skipped+=1;}else{parents[x]=y;sizes[y]+=sizes[x];pairs.push(Pair{a,b,ab});}}
        let selected_edges=pairs.iter().map(|p|(p.a,p.b)).collect();
        Ok(Self{tree:Tree::factor(&diag,&pairs)?,free:matrix.free.clone(),bodies:matrix.body_count,selected_edges,omitted_cycles:skipped})
    }
    /// Maximum normalized-weight forest over every nonzero free-body block.
    pub fn from_dense_weighted(&self,columns:&[f32])->Result<Self,Unsupported>{
        let n=self.free.len();let m=n.checked_mul(6).ok_or_else(||Unsupported("dense extent overflow".into()))?;
        if columns.len()!=m.checked_mul(m).ok_or_else(||Unsupported("dense extent overflow".into()))? || columns.iter().any(|x|!x.is_finite()){return Err(Unsupported("dense input invalid".into()));}
        let mut diag=vec![[[0f64;6];6];n];let mut ls=diag.clone();
        for a in 0..n{for i in 0..6{for j in 0..6{let r=6*a+i;let c=6*a+j;diag[a][i][j]=0.5*((columns[c*m+r] as f64)+(columns[r*m+c] as f64));}}
            for i in 0..6{for j in 0..=i{let mut v=diag[a][i][j];for k in 0..j{v-=ls[a][i][k]*ls[a][j][k];}if i==j{if !v.is_finite()||v<=0.{return Err(Unsupported("nonpositive diagonal pivot".into()));}ls[a][i][j]=v.sqrt();}else{ls[a][i][j]=v/ls[a][j][j];}}}}
        let mut candidates=Vec::new();for a in 0..n{for b in a+1..n{let mut c=[[0f64;6];6];for i in 0..6{for j in 0..6{let r=6*a+i;let s=6*b+j;c[i][j]=0.5*((columns[s*m+r] as f64)+(columns[r*m+s] as f64));}}
            if c.iter().flatten().all(|v|*v==0.){continue;}
            let mut x=c;for col in 0..6{for i in 0..6{for k in 0..i{x[i][col]-=ls[a][i][k]*x[k][col];}x[i][col]/=ls[a][i][i];}}
            let mut y=x;for row in 0..6{for i in 0..6{for k in 0..i{y[row][i]-=ls[b][i][k]*y[row][k];}y[row][i]/=ls[b][i][i];}}
            let w=y.iter().flatten().map(|v|v*v).sum::<f64>().sqrt();if !w.is_finite(){return Err(Unsupported("nonfinite edge weight".into()));}candidates.push((w,a,b,c));}}
        candidates.sort_by(|x,y|y.0.total_cmp(&x.0).then_with(||self.free[x.1].cmp(&self.free[y.1])).then_with(||self.free[x.2].cmp(&self.free[y.2])));
        let mut parents:Vec<_>=(0..n).collect();fn find(p:&mut [usize],mut x:usize)->usize{while p[x]!=x{p[x]=p[p[x]];x=p[x];}x}
        let mut pairs=Vec::new();let mut skipped=0;for (_,a,b,ab) in candidates {let x=find(&mut parents,a);let y=find(&mut parents,b);if x==y{skipped+=1;}else{parents[x]=y;pairs.push(Pair{a,b,ab});}}
        let selected_edges=pairs.iter().map(|p|(p.a,p.b)).collect();Ok(Self{tree:Tree::factor(&diag,&pairs)?,free:self.free.clone(),bodies:self.bodies,selected_edges,omitted_cycles:skipped})
    }
    /// Same forest topology, blocks sampled from actual projected A_bb.
    pub fn from_dense(&self, columns:&[f32])->Result<Self,Unsupported>{
        let n=self.free.len();let m=n.checked_mul(6).ok_or_else(||Unsupported("dense extent overflow".into()))?;
        if columns.len()!=m.checked_mul(m).ok_or_else(||Unsupported("dense extent overflow".into()))? || columns.iter().any(|x|!x.is_finite()) {return Err(Unsupported("dense block input invalid".into()));}
        let mut diag=vec![[[0f64;6];6];n];
        for a in 0..n {for i in 0..6 {for j in 0..6 {let r=6*a+i;let c=6*a+j;diag[a][i][j]=0.5*((columns[c*m+r] as f64)+(columns[r*m+c] as f64));}}}
        let mut pairs=Vec::new();for &(a,b) in &self.selected_edges {let mut ab=[[0f64;6];6];for i in 0..6 {for j in 0..6 {let r=6*a+i;let c=6*b+j;ab[i][j]=0.5*((columns[c*m+r] as f64)+(columns[r*m+c] as f64));}}pairs.push(Pair{a,b,ab});}
        Ok(Self{tree:Tree::factor(&diag,&pairs)?,free:self.free.clone(),bodies:self.bodies,selected_edges:self.selected_edges.clone(),omitted_cycles:self.omitted_cycles})
    }
    pub fn free_indices(&self)->Vec<usize>{self.free.iter().flat_map(|&b|(0..6).map(move |k|6*b+k)).collect()}
    pub fn apply(&self,rhs:&[f64])->Result<Vec<f64>,Unsupported>{
        if self.bodies.checked_mul(6)!=Some(rhs.len()) {return Err(Unsupported("body rhs length".into()));}
        let reduced:Vec<_>=self.free.iter().flat_map(|&b|rhs[6*b..6*b+6].iter().copied()).collect();
        let x=self.tree.apply(&reduced)?;let mut out=vec![0.;rhs.len()];
        for (i,&b) in self.free.iter().enumerate(){out[6*b..6*b+6].copy_from_slice(&x[6*i..6*i+6]);} Ok(out)
    }
    /// Preserve the original nodal BJ output and replace only body entries.
    pub fn apply_mixed(&self,rhs:&[f32],bj:&[f32],base:usize)->Result<Vec<f32>,Unsupported>{
        let end=base.checked_add(self.bodies.checked_mul(6).ok_or_else(||Unsupported("body extent overflow".into()))?).ok_or_else(||Unsupported("mixed extent overflow".into()))?;
        if rhs.len()!=end||bj.len()!=end{return Err(Unsupported("mixed extent mismatch".into()));}
        let body:Vec<_>=rhs[base..].iter().map(|v|*v as f64).collect();let solved=self.apply(&body)?;
        let mut out=bj.to_vec();for (dst,v) in out[base..].iter_mut().zip(solved){*dst=v as f32;if !dst.is_finite(){return Err(Unsupported("mixed result nonfinite".into()));}} Ok(out)
    }
}

/// Host shadow on one immutable projected operator. `r0` is the real affine
/// seed residual; `verify` computes the original true residual of x0+delta.
pub enum Call { Apply, Precond, Verify }
pub fn shadow_pcg<C>(r0:&[f32], initial:&[f32], body_base:usize, tol:f32, cap:u32,
    mut call:C)->Result<(u32,f32),String>
where C:FnMut(Call,&[f32])->Result<(Vec<f32>,f32),String> {
    let dim=r0.len(); if body_base>dim || body_base%3!=0 || (dim-body_base)%6!=0 || initial.len()!=1+(dim-body_base)/6 || cap==0 {return Err("invalid shadow dimensions".into())}
    if !tol.is_finite() || tol<=0. || r0.iter().any(|v|!v.is_finite()) || initial.iter().any(|v|!v.is_finite() || *v<0.) || !initial.iter().sum::<f32>().is_finite() {return Err("invalid shadow inputs".into());}
    let mut x=vec![0f32;dim];
    if r0.iter().all(|v|*v==0.) {let (r,v)=call(Call::Verify,&x)?;if r.len()!=dim || r.iter().any(|v|!v.is_finite()) || !v.is_finite() || v<0. {return Err("invalid initial true check".into())} if v<tol {return Ok((0,v));} return Err("zero initial residual failed true check".into());}
    if initial.iter().sum::<f32>()<=0. {return Err("nonzero residual with zero initial scale".into());}
    let mut r=r0.to_vec();let mut p=vec![0f32;dim];let mut z=call(Call::Precond,&r)?.0;
    if z.len()!=dim || z.iter().any(|v|!v.is_finite()){return Err("invalid preconditioner output".into())} p.copy_from_slice(&z);
    let dot=|a:&[f32],b:&[f32]|->f64{a.iter().zip(b).map(|(a,b)|(*a as f64)*(*b as f64)).sum()};
    let mut rz=dot(&r,&z); if !rz.is_finite() || rz<=0. {return Err("invalid initial rMz".into())}
    let scale: f32=initial.iter().sum();
    for it in 1..=cap {
        let ap=call(Call::Apply,&p)?.0;if ap.len()!=dim || ap.iter().any(|v|!v.is_finite()){return Err("invalid operator output".into())}
        let pap=dot(&p,&ap);if !pap.is_finite() || pap<=0. {return Err("nonpositive pAp".into())}
        let alpha=(rz/pap) as f32;if !alpha.is_finite(){return Err("invalid alpha".into())} for i in 0..dim{x[i]+=alpha*p[i];r[i]-=alpha*ap[i];} if x.iter().chain(&r).any(|v|!v.is_finite()){return Err("invalid iterate".into())}
        let mut norms=vec![0f32;initial.len()];norms[0]=r[..body_base].iter().map(|v|v.abs()).sum();
        for j in 1..norms.len(){norms[j]=r[body_base+(j-1)*6..body_base+j*6].iter().map(|v|v.abs()).sum();}
        let relative=norms.iter().zip(initial).map(|(n,s)|{let ratio=n / if *s>0. {*s} else {scale};if ratio.is_finite(){ratio}else{f32::INFINITY}}).fold(0f32,f32::max);
        if relative<tol {let (true_r,verified)=call(Call::Verify,&x)?;if true_r.len()!=dim || true_r.iter().any(|v|!v.is_finite()) || !verified.is_finite() || verified<0. {return Err("invalid true residual".into())} if verified<tol{return Ok((it,verified))}r=true_r;z=call(Call::Precond,&r)?.0;if z.len()!=dim || z.iter().any(|v|!v.is_finite()){return Err("invalid preconditioner output".into())}p.copy_from_slice(&z);rz=dot(&r,&z);}
        else {z=call(Call::Precond,&r)?.0;let next=dot(&r,&z);if z.len()!=dim || z.iter().any(|v|!v.is_finite()){return Err("invalid preconditioner output".into())} let beta=(next/rz) as f32;if !beta.is_finite(){return Err("invalid beta".into())} for i in 0..dim{p[i]=z[i]+beta*p[i];}if p.iter().any(|v|!v.is_finite()){return Err("invalid direction".into())}rz=next;}
        if !rz.is_finite() || rz<=0. {return Err("invalid rMz during shadow".into())}
    } Err("shadow iteration cap".into())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn lagged_domain_rejects_reordered_body_and_root(){let d=ShadowDomain{version:1,scene_uuid:"scene".into(),scene_seal:"seal".into(),nrow:2,ncloth:0,bodies:2,dim:12,body_base:0,vbody:vec![0,1],cloth_off:vec![0,0],free:(6..12).collect(),projection:false,lock:false,hard:false};let path=std::env::temp_dir().join(format!("ppf_lagged_domain_test_{}.json",std::process::id()));let record=LaggedRecord{domain:d,columns:vec![1.],source_solve:1,start_time:1.,sample_time:1.};std::fs::write(&path,serde_json::to_vec(&record).unwrap()).unwrap();let mut expected:ShadowDomain=serde_json::from_value(serde_json::to_value(&record.domain).unwrap()).unwrap();assert!(load_lagged(&path,&expected).is_ok());expected.vbody.swap(0,1);assert!(load_lagged(&path,&expected).is_err());expected.vbody.swap(0,1);expected.free=(0..6).collect();assert!(load_lagged(&path,&expected).is_err());std::fs::remove_file(path).unwrap();}
    fn diagonal(v:f64)->Block {let mut b=[[0.;6];6];for i in 0..6{b[i][i]=v;}b}
    fn edge(a:usize,b:usize,v:f64)->Pair {Pair{a,b,ab:diagonal(v)}}
    #[test] fn sparse_weighted_matches_dense_with_loop_and_missing_reverse() {
        use super::super::body_block_projection::SparseBodyMatrix;
        let n=4;let m=n*6;let mut blocks=BTreeMap::new();
        for a in 0..n {let mut d=[[0f32;6];6];for i in 0..6{d[i][i]=4.;}blocks.insert((a,a),d);}
        for (a,b,v) in [(0,1,-0.6),(1,2,-0.4),(2,0,-0.2),(2,3,-0.1)] {let mut x=[[0f32;6];6];for i in 0..6{x[i][i]=v;}blocks.insert((a,b),x);}
        let sm=SparseBodyMatrix{body_count:n,free:(0..n).collect(),blocks};let mut dense=vec![0f32;m*m];
        for (&(a,b),block) in &sm.blocks {for i in 0..6{for j in 0..6{dense[(6*b+j)*m+6*a+i]=block[i][j];}}}
        let seed=prepare(&vec![diagonal(1.);n],&vec![false;n],&[]).unwrap();let old=seed.from_dense_weighted(&dense).unwrap();let new=FreeForest::from_sparse_weighted(&sm,170).unwrap();
        assert_eq!(old.selected_edges,new.selected_edges);assert_eq!(old.tree.roots,new.tree.roots);
        let rhs:Vec<f64>=(0..m).map(|i|i as f64/13.).collect();assert_eq!(old.apply(&rhs).unwrap(),new.apply(&rhs).unwrap());
    }
    #[test] fn tree_centers_match_minimum_eccentricity_ties_and_components() {
        let d=vec![diagonal(5.);9];
        let edges=vec![edge(0,1,-0.1),edge(1,2,-0.1),edge(2,3,-0.1),edge(4,5,-0.1),edge(4,6,-0.1),edge(4,7,-0.1)];
        let tree=Tree::factor(&d,&edges).unwrap();
        assert_eq!(tree.roots,vec![1,4,8]);
    }
    #[test] fn sparse_large_chain_is_partitioned_and_spd() {
        use super::super::body_block_projection::SparseBodyMatrix;
        let n=171;let mut blocks=BTreeMap::new();
        for a in 0..n {let mut d=[[0f32;6];6];for i in 0..6{d[i][i]=4.;}blocks.insert((a,a),d);if a>0{let mut e=[[0f32;6];6];for i in 0..6{e[i][i]=-0.5;}blocks.insert((a-1,a),e);blocks.insert((a,a-1),e);}}
        let sm=SparseBodyMatrix{body_count:n,free:(0..n).collect(),blocks};let forest=FreeForest::from_sparse_weighted(&sm,170).unwrap();
        assert_eq!(forest.tree.roots.len(),2);assert_eq!(forest.selected_edges.len(),169);
        let rhs=vec![1.;6*n];let x=forest.apply(&rhs).unwrap();assert!(x.iter().all(|v|v.is_finite()));assert!(x.iter().zip(rhs).map(|(a,b)|a*b).sum::<f64>()>0.);
    }
    #[test] fn cycle_selects_forest_and_solves(){
        let t=prepare(&[diagonal(4.);3],&[false;3],&[edge(0,1,-1.),edge(1,2,-1.),edge(0,2,-1.)]).unwrap();
        assert_eq!(t.omitted_cycles,1);let x=t.apply(&[1.;18]).unwrap();
        for k in 0..6 {assert!((4.*x[k]-x[6+k]-x[12+k]-1.).abs()<1e-10);assert!((4.*x[6+k]-x[k]-1.).abs()<1e-10);assert!((4.*x[12+k]-x[k]-1.).abs()<1e-10);}
    }
    #[test] fn fixed_body_eliminated_self_diagonal_kept(){
        let t=prepare(&[diagonal(4.);2],&[true,false],&[edge(0,1,-1.)]).unwrap();
        let x=t.apply(&[1.;12]).unwrap();assert!(x[..6].iter().all(|&v|v==0.));assert!(x[6..].iter().all(|&v|(v-0.25).abs()<1e-10));
    }
    #[test] fn indefinite_schur_refuses(){assert!(prepare(&[diagonal(1.);2],&[false;2],&[edge(0,1,-2.)]).is_err());}
    #[test] fn mixed_prefix_and_middle_root_mapping(){let t=prepare(&[diagonal(2.),diagonal(3.),diagonal(4.)],&[false,true,false],&[]).unwrap();let mut r=vec![1.;21];r[..3].copy_from_slice(&[2.,3.,4.]);let z=t.apply_mixed(&r,&[9.;21],3).unwrap();assert_eq!(&z[..3],&[9.,9.,9.]);assert!(z[3..9].iter().all(|v|*v==0.5));assert!(z[9..15].iter().all(|v|*v==0.));assert!(z[15..].iter().all(|v|*v==0.25));}
    #[test] fn shadow_uses_true_check(){let mut checks=0;let result=shadow_pcg(&[2.;6],&[0.,12.],0,1e-5,8,|op,x|match op{Call::Apply=>Ok((x.iter().map(|x|x*2.).collect(),0.)),Call::Precond=>Ok((x.to_vec(),0.)),Call::Verify=>{checks+=1;let r:Vec<_>=x.iter().map(|v|2.-2.*v).collect();let e=r.iter().map(|v|v.abs()).sum::<f32>()/12.;Ok((r,e))}}).unwrap();assert_eq!(result.0,1);assert_eq!(checks,1);}
}
