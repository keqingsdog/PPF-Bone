//! Stable device coalescing of complete reduced CSR; numerical coefficients are
//! refreshed every frozen Newton. Exact CSR+mapping comparison guards reuse.
use ppf_cts_compute::{AllocLabel,Buffer,Device,Fault,Handle};
use super::{kernels::*,devsort::SortScratch,scan::ScanScratch,reduced_model::Revision};
fn refuse(s:impl Into<String>)->Fault{Fault::Platform{call:"reduced CSR coalescing",detail:s.into()}}
#[derive(Clone,Copy)]
pub struct View{pub offset:Handle,pub column:Handle,pub value:Handle,pub blocks:u32}
#[derive(Clone,Copy,Default,Debug)]
pub struct Stats{pub raw_blocks:u32,pub unique_blocks:u32,pub topology_cache_hit:bool,
 pub topology_ms:f64,pub numeric_ms:f64,pub total_ms:f64,pub edge_weight_ms:f64}
#[derive(Default)]
pub struct Coalescer{
 row_id:Buffer<u32>,row_key:Buffer<u32>,permutation:Buffer<u32>,flags:Buffer<u32>,
 run_start:Buffer<u32>,run_length:Buffer<u32>,offset:Buffer<u32>,column:Buffer<u32>,value:Buffer<f32>,
 cached_offset:Buffer<u32>,cached_column:Buffer<u32>,cached_body:Buffer<u32>,cached_free:Buffer<u32>,
 status:Buffer<u32>,dummy:Buffer<f32>,column_sort:SortScratch,row_sort:SortScratch,scan:ScanScratch,
 shape:Option<(u32,u32,u32,u32,u32)>,ready:Option<Revision>,unique:u32,
 graph_offset:Vec<u32>,graph_column:Vec<u32>,graph_weight:Vec<u32>,structural_weight:Vec<u32>,
 diagonal_strength:Buffer<f32>,edge_strength:Buffer<f32>,
 diagnostic_done:bool,diagnostic_revision:Option<Revision>,stats:Stats,
}
unsafe fn read_u32<D:Device>(d:&mut D,h:Handle,n:usize)->Result<Vec<u32>,Fault>{
 let mut x=vec![0u32;n];d.read(h,0,std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(),4*n))?;Ok(x)
}
unsafe fn read_f32<D:Device>(d:&mut D,h:Handle,n:usize)->Result<Vec<f32>,Fault>{
 let mut x=vec![0f32;n];d.read(h,0,std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(),4*n))?;Ok(x)
}
impl Coalescer{
 pub fn view(&self,current:Revision)->Option<View>{(self.ready==Some(current)).then(||View{
  offset:self.offset.handle(),column:self.column.handle(),value:self.value.handle(),blocks:self.unique})}
 pub fn diagnostic_revision(&self)->Option<Revision>{self.diagnostic_revision}
 pub fn stats(&self)->Stats{self.stats}
 pub fn graph(&self)->(&[u32],&[u32],&[u32]){(&self.graph_offset,&self.graph_column,&self.graph_weight)}
 fn allocate<D:Device>(&mut self,d:&mut D,rows:u32,blocks:u32,vertices:u32)->Result<(),Fault>{
  let n=(blocks as usize).max(1);let v=(vertices as usize).max(1);
  self.row_id.size(d,n,AllocLabel("rsc.row_id"))?;self.row_key.size(d,n,AllocLabel("rsc.row_key"))?;
  self.permutation.size(d,n,AllocLabel("rsc.permutation"))?;self.flags.size(d,n+1,AllocLabel("rsc.head_scan"))?;
  if self.run_start.len()==0{self.run_start.size(d,1,AllocLabel("rsc.run_start"))?;}
  if self.run_length.len()==0{self.run_length.size(d,1,AllocLabel("rsc.run_length"))?;}
  self.offset.size(d,rows as usize+1,AllocLabel("rsc.unique_offset"))?;
  if self.column.len()==0{self.column.size(d,1,AllocLabel("rsc.unique_column"))?;}
  if self.value.len()==0{self.value.size(d,1,AllocLabel("rsc.unique_value"))?;}
  self.cached_offset.size(d,rows as usize+1,AllocLabel("rsc.cached_offset"))?;
  self.cached_column.size(d,n,AllocLabel("rsc.cached_column"))?;
  self.cached_body.size(d,v,AllocLabel("rsc.cached_vertex_body"))?;
  self.cached_free.size(d,v,AllocLabel("rsc.cached_vertex_free"))?;
  self.status.size(d,2,AllocLabel("rsc.private_status"))?;self.dummy.size(d,1,AllocLabel("rsc.inactive_float"))?;
  Ok(())
 }
 fn prep(&self,offset:Handle,column:Handle,body:Handle,free:Handle,rows:u32,blocks:u32,vertices:u32,mode:u32,extent:u32)->RscPrepareArgs{
  RscPrepareArgs{offset,column,permutation:self.column_sort.index.handle(),vertex_body:body,vertex_free:free,
   cached_offset:self.cached_offset.handle(),cached_column:self.cached_column.handle(),cached_body:self.cached_body.handle(),cached_free:self.cached_free.handle(),
   row_id:self.row_id.handle(),row_key:self.row_key.handle(),status:self.status.handle(),rows,blocks,vertices,mode,extent,count:extent,seam_arena_count:0}
 }
 fn reduce(&self,raw_offset:Handle,raw_column:Handle,raw_value:Handle,rows:u32,blocks:u32,mode:u32,extent:u32)->RscReduceArgs{
  RscReduceArgs{raw_offset,raw_column,raw_value,row_id:self.row_id.handle(),permutation:self.permutation.handle(),scanned:self.flags.handle(),
   run_start:self.run_start.handle(),run_length:self.run_length.handle(),unique_count:self.offset.handle(),unique_offset:self.offset.handle(),
   unique_column:self.column.handle(),unique_value:self.value.handle(),input:self.dummy.handle(),old_product:self.dummy.handle(),new_product:self.dummy.handle(),
   status:self.status.handle(),rows,blocks,unique_blocks:self.unique,mode,extent,count:extent,seam_arena_count:0}
 }
 unsafe fn status<D:Device>(&self,d:&mut D)->Result<bool,Fault>{
  let status=read_u32(d,self.status.handle(),2)?;
  if status[1]!=0{return Err(refuse(format!("sticky device fault count {}",status[1])));}
  Ok(status[0]!=0)
 }
 pub unsafe fn rebuild<D:Device>(&mut self,d:&mut D,raw_offset:Handle,raw_column:Handle,raw_value:Handle,
  body:Handle,free:Handle,rows:u32,blocks:u32,vertices:u32,cloth:u32,bodies:u32,current:Revision)->Result<(),Fault>{
  self.ready=None;let started=std::time::Instant::now();
  if rows==0||blocks==0||blocks>u32::MAX/9{return Err(refuse("nonempty CSR and checked scalar capacity required"));}
  self.allocate(d,rows,blocks,vertices)?;d.fill_zero(self.status.handle(),8)?;
  let shape=(rows,blocks,vertices,cloth,bodies);let mut hit=false;
  if self.shape==Some(shape){
   let n=(rows+1).max(blocks).max(vertices);let a=self.prep(raw_offset,raw_column,body,free,rows,blocks,vertices,2,n);
   d.launch("rsc.exact_topology_mapping_compare",&a,n)?;hit=!self.status(d)?;
  }
  let topology=std::time::Instant::now();
  if !hit{
   self.shape=None;self.unique=0;
   // mode0 never dereferences permutation; nevertheless its pointer must be a
   // valid arena handle before the generated entry resolves every argument.
   if self.column_sort.index.len()==0{self.column_sort.index.size(d,1,AllocLabel("rsc.inactive_permutation"))?;}
   let a=self.prep(raw_offset,raw_column,body,free,rows,blocks,vertices,0,rows);
   d.launch("rsc.rows",&a,rows)?;self.status(d)?;
   self.column_sort.sort(d,"rsc.stable_column_sort",raw_column,blocks as usize)?;
   let a=self.prep(raw_offset,raw_column,body,free,rows,blocks,vertices,1,blocks);
   d.launch("rsc.row_keys",&a,blocks)?;self.status(d)?;
   self.row_sort.sort(d,"rsc.stable_row_sort",self.row_key.handle(),blocks as usize)?;
   d.fill_zero(self.flags.handle(),(blocks as usize+1)*4)?;
   let a=RscSegmentsArgs{column:raw_column,row_id:self.row_id.handle(),column_permutation:self.column_sort.index.handle(),
    row_permutation:self.row_sort.index.handle(),permutation:self.permutation.handle(),flags:self.flags.handle(),status:self.status.handle(),
    rows,extent:blocks,count:blocks,seam_arena_count:0};
   d.launch("rsc.unique_heads",&a,blocks)?;self.status(d)?;
   self.scan.exclusive(d,"rsc.scan_heads",self.flags.handle(),blocks)?;
   let mut total=[0u8;4];d.read(self.flags.handle(),blocks as usize*4,&mut total)?;self.unique=u32::from_le_bytes(total);
   if self.unique==0||self.unique>blocks{return Err(refuse("unique total outside exact raw extent"));}
   self.run_start.size(d,self.unique as usize,AllocLabel("rsc.run_start"))?;
   self.run_length.size(d,self.unique as usize,AllocLabel("rsc.run_length"))?;
   self.column.size(d,self.unique as usize,AllocLabel("rsc.unique_column"))?;
   self.value.size(d,self.unique as usize*9,AllocLabel("rsc.unique_value"))?;
   d.fill_zero(self.offset.handle(),(rows as usize+1)*4)?;
   let a=self.reduce(raw_offset,raw_column,raw_value,rows,blocks,0,blocks);
   d.launch("rsc.run_topology",&a,blocks)?;self.status(d)?;
   self.scan.exclusive(d,"rsc.scan_unique_rows",self.offset.handle(),rows)?;
   let mut total=[0u8;4];d.read(self.offset.handle(),rows as usize*4,&mut total)?;
   if u32::from_le_bytes(total)!=self.unique{return Err(refuse("unique row scan did not cover all runs"));}
   // Publish exact key cache only after all topology construction guards pass.
   d.copy(self.cached_offset.handle(),0,raw_offset,0,(rows as usize+1)*4)?;
   d.copy(self.cached_column.handle(),0,raw_column,0,blocks as usize*4)?;
   if vertices!=0{d.copy(self.cached_body.handle(),0,body,0,vertices as usize*4)?;
    d.copy(self.cached_free.handle(),0,free,0,vertices as usize*4)?;}
   self.graph_offset=read_u32(d,self.offset.handle(),rows as usize+1)?;
   self.graph_column=read_u32(d,self.column.handle(),self.unique as usize)?;
   self.structural_weight=read_u32(d,self.run_length.handle(),self.unique as usize)?;
   if self.structural_weight.iter().map(|&x|u64::from(x)).sum::<u64>()!=u64::from(blocks)||self.structural_weight.iter().any(|&x|x==0){return Err(refuse("run multiplicities do not cover every raw block"));}
   for r in 0..rows as usize{let span=&self.graph_column[self.graph_offset[r]as usize..self.graph_offset[r+1]as usize];
    if span.windows(2).any(|p|p[0]>=p[1])||span.iter().any(|&c|c>=rows){return Err(refuse("coalesced CSR not strictly unique/sorted"));}}
   self.shape=Some(shape);
  }
  let topology_ms=topology.elapsed().as_secs_f64()*1000.;let numeric=std::time::Instant::now();
  // CUDA mode1 interprets one logical Args item as one complete warp/run.
  // Keep extent/count/device.launch count in LOGICAL RUNS (the seam contract),
  // never inflate count by32. Custom launcher uses ceil(runs/4) CTAs of128.
  let logical_runs=self.unique;
  if logical_runs==0||logical_runs>blocks{return Err(refuse("invalid warp numeric logical run count"));}
  let ctas=u64::from(logical_runs/4)+u64::from((logical_runs%4!=0)as u32);
  let physical_threads=ctas.checked_mul(128).ok_or_else(||refuse("warp launch physical thread overflow"))?;
  if ctas>i32::MAX as u64||physical_threads<32*u64::from(logical_runs){return Err(refuse("warp launch grid extent invalid"));}
  let a=self.reduce(raw_offset,raw_column,raw_value,rows,blocks,1,logical_runs);
  d.launch("rsc.refresh_numeric_warp32",&a,logical_runs)?;self.status(d)?;
  let numeric_ms=numeric.elapsed().as_secs_f64()*1000.;
  let weight_started=std::time::Instant::now();
  let weight_mode=std::env::var("PPF_SCHWARZ_EDGE_WEIGHT").unwrap_or_else(|_|"structural".into());
  match weight_mode.as_str(){
   "structural"=>self.graph_weight=self.structural_weight.clone(),
   "numeric"=>{
    self.diagonal_strength.size(d,rows as usize,AllocLabel("rsc.diagonal_strength"))?;
    self.edge_strength.size(d,self.unique as usize,AllocLabel("rsc.edge_strength"))?;
    let mut a=self.reduce(raw_offset,raw_column,raw_value,rows,blocks,3,rows);
    a.old_product=self.diagonal_strength.handle();a.new_product=self.edge_strength.handle();
    d.launch("rsc.edge_diagonal",&a,rows)?;
    a.mode=4;
    d.launch("rsc.edge_strength",&a,rows)?;
    self.status(d)?;
    let strength=read_f32(d,self.edge_strength.handle(),self.unique as usize)?;
    self.graph_weight=strength.into_iter().map(|x|{
     // Fixed positive quantization gives the topology cache a stable exact key.
     // The weight is only a matching heuristic, never a physical coefficient.
     ((x.min(1000.)*1_000_000.).round() as u32).max(1)
    }).collect();
   },
   _=>return Err(refuse("PPF_SCHWARZ_EDGE_WEIGHT must be structural or numeric")),
  }
  let edge_weight_ms=weight_started.elapsed().as_secs_f64()*1000.;
  self.stats=Stats{raw_blocks:blocks,unique_blocks:self.unique,topology_cache_hit:hit,topology_ms,numeric_ms,total_ms:started.elapsed().as_secs_f64()*1000.,edge_weight_ms};
  if !self.diagnostic_done&&std::env::var_os("PPF_SCHWARZ_FROZEN_CHECK").is_some(){
   if std::env::var("PPF_SCHWARZ_FROZEN_CHECK").ok().as_deref()!=Some("1"){return Err(refuse("diagnostic flag must be exactly1"));}
   self.diagnostic(d,raw_offset,raw_column,raw_value,rows,blocks,current)?;self.diagnostic_revision=Some(current);self.diagnostic_done=true;
  }
  self.ready=Some(current);Ok(())
 }
 unsafe fn diagnostic<D:Device>(&mut self,d:&mut D,raw_offset:Handle,raw_column:Handle,raw_value:Handle,rows:u32,blocks:u32,current:Revision)->Result<(),Fault>{
  let dim=3*rows as usize;if dim>512||blocks>1_000_000{return Err(refuse("opt-in coalescing diagnostic requires dim<=512/raw<=1M"));}
  let ro=read_u32(d,raw_offset,rows as usize+1)?;let rc=read_u32(d,raw_column,blocks as usize)?;let rv=read_f32(d,raw_value,9*blocks as usize)?;
  let uv=read_f32(d,self.value.handle(),9*self.unique as usize)?;
  let mut x=Buffer::<f32>::default();let mut old=Buffer::<f32>::default();let mut new=Buffer::<f32>::default();
  x.size(d,dim,AllocLabel("rsc.check.x"))?;old.size(d,dim,AllocLabel("rsc.check.old_product"))?;new.size(d,dim,AllocLabel("rsc.check.new_product"))?;
  let mut worst=0f64;
  for probe in 0..dim+2{let mut v=vec![0f32;dim];if probe<dim{v[probe]=1.;}else{for i in 0..dim{v[i]=if probe==dim{((7*i)%11)as f32/8.-5./8.}else{((5*i)%13)as f32/16.-6./16.};}}x.write(d,0,&v)?;
   let mut a=self.reduce(raw_offset,raw_column,raw_value,rows,blocks,2,rows);a.input=x.handle();a.old_product=old.handle();a.new_product=new.handle();
   d.launch("rsc.check_actual_old_new_products",&a,rows)?;self.status(d)?;
   let old_gpu=read_f32(d,old.handle(),dim)?;let new_gpu=read_f32(d,new.handle(),dim)?;
   for row in 0..rows as usize{for r in 0..3{let mut od=0f64;let mut ud=0f64;let mut abs=0f64;
    for k in ro[row]as usize..ro[row+1]as usize{for c in 0..3{let q=f64::from(rv[9*k+3*c+r])*f64::from(v[3*rc[k]as usize+c]);od+=q;abs+=q.abs();}}
    for k in self.graph_offset[row]as usize..self.graph_offset[row+1]as usize{for c in 0..3{ud+=f64::from(uv[9*k+3*c+r])*f64::from(v[3*self.graph_column[k]as usize+c]);}}
    // Gamma bound accounts the actual longest componentwise raw accumulation;
    // it is preregistered arithmetic roundoff, never the solver tolerance.
    let n=3*(ro[row+1]-ro[row])as usize+1;let ne=n as f64*f64::from(f32::EPSILON);
    if ne>=0.25{return Err(refuse("diagnostic accumulation bound ambiguous"));}
    let bound=8.*ne/(1.-ne)*abs;
    for error in [(od-ud).abs(),(od-f64::from(old_gpu[3*row+r])).abs(),(ud-f64::from(new_gpu[3*row+r])).abs()]{
     let ratio=if bound==0.{if error==0.{0.}else{f64::INFINITY}}else{error/bound};worst=worst.max(ratio);
     if !ratio.is_finite()||ratio>1.{return Err(refuse(format!("old/new product mismatch probe{probe} row{row} component{r}: {error:e}>{bound:e}")));}
    }
   }}
  }
  eprintln!("SCHWARZ_COALESCE_CHECK {}",serde_json::json!({"schema":"ppf.csr-coalesce-check.v1","same_frozen_revision":current.operator,"mapping_revision":current.mapping,"joint_revision":current.joints,"rows":rows,"raw_blocks":blocks,"unique_blocks":self.unique,"basis_columns":dim,"mixed_probes":2,"max_component_error_ratio":worst,"zero_cancel_blocks_retained":true,"duplicate_run_order":"32 striped original-index partial sums then fixed float32 pairwise offsets16,8,4,2,1; CPU and CUDA same algebra","numeric_reduction_contract":"each lane adds original positions lane+32*t; fixed balanced32 tree; no sequential-bitwise claim",
   "comparison":"actual GPU old/new products + independent CPU FP64 old/new CSR sums","bound":"8*gamma_(3*raw_row_blocks+1)*sumabs(raw component terms); reject n*eps>=.25; zero exact"}));Ok(())
 }
}

