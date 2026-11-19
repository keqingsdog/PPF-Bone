//! Concrete Stage10 device adapter. All numerical B/factors stay on device.
//! Graph offsets/column IDs and explicit fault/regularization metadata may be read back.
use ppf_cts_compute::{AllocLabel, Buffer, Device, Encoder, EncoderExt, Fault, Handle, ReadbackBuffer};
use super::{kernels::*, reduced_model::{self as model, Backend, Domains, Layout, Revision},
    reduced_schwarz::FrozenJointBlockView, scan::ScanScratch, schwarz::OperatorRows};

fn refusal(detail: impl Into<String>) -> Fault {
    Fault::Platform { call: "full reduced Schwarz", detail: detail.into() }
}

#[derive(Clone, Copy)]
pub struct Inputs {
    pub rows: OperatorRows,
    /// Actual inverse free map, one vertex ID per cloth triplet.
    pub row_to_vertex: Handle,
    /// Valid free row or NONE for every full-space vertex.
    pub vertex_to_free: Handle,
    /// Original reduction metadata and current R*ybar, full-vertex indexed.
    pub vertex_body: Handle,
    pub rotated_rest: Handle,
    pub body_state: Handle,
    pub source_rows: u32,
    pub dt: f32,
    /// Row-major36/body, before material additions and before BJ factorization.
    pub raw_body: Handle,
    pub layout: Layout,
    pub constraints:super::constraint_space::View,
}

#[derive(Clone, Copy)]
pub struct ModelView {
    pub rows: u32,
    pub blocks: u32,
    pub offset: Handle,
    pub column: Handle,
    pub value: Handle,
    pub revision: Revision,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    pub rows: u32,
    pub blocks: u32,
    pub raw_blocks:u32,
    pub coalesce_cache_hit:bool,
    pub coalesce_topology_ms:f64,
    pub coalesce_numeric_ms:f64,
    pub coalesce_total_ms:f64,
    pub edge_weight_ms:f64,
    pub partition_reused:bool,
    pub partition_ms:f64,
    pub domains: u32,
    pub body_shifted: u32,
    pub body_shift_max: f32,
    pub local_jittered: u32,
    pub local_jitter_max: f32,
}

#[derive(Clone,Copy,Debug,Default)]
pub struct ApplyReceipt {
    pub dispatches: u32,
    pub domain_successes: u32,
    pub domain_faults: u32,
}

#[derive(Default)]
struct Storage {
    offset: ReadbackBuffer<u32>,
    cursor: Buffer<u32>,
    base_count: Buffer<u32>,
    projection_fault: ReadbackBuffer<u32>,
    column: ReadbackBuffer<u32>,
    value: Buffer<f32>,
    body_seed: Buffer<f32>,
    body_inertia: Buffer<f32>,
    inactive_body_state: Buffer<crate::data::PdrdRigidState>,
    body_shift: ReadbackBuffer<f32>,
    fault: ReadbackBuffer<u32>,
    receipt: ReadbackBuffer<u32>,
    member_offset: Buffer<u32>,
    members: Buffer<u32>,
    assignment: Buffer<u32>,
    dense: Buffer<f32>,
    normalized: Buffer<f32>,
    lower: Buffer<f32>,
    inverse: Buffer<f32>,
    scale: Buffer<f32>,
    shift: ReadbackBuffer<f32>,
    scan: ScanScratch,
    capacity: usize,
    // Exact used structural data copied at graph() before any later dispatch.
    graph_offset: Vec<u32>,
    graph_column: Vec<u32>,
    graph_weight: Vec<u32>,
    coalescer: super::csr_coalesce::Coalescer,
    stats: Stats,
}

#[derive(Default)]
pub struct ReducedModel {
    core: model::State,
    storage: Storage,
    ready: Option<Revision>,
    graph_revision: Option<Revision>,
}

impl ReducedModel {
    /// Rebuild at the exact frozen Newton context. Inputs must be live and satisfy
    /// original operator/map extents. No count-only numerical cache reuse.
    pub unsafe fn rebuild<D: Device>(&mut self, device: &mut D, inputs: Inputs,
        joints: FrozenJointBlockView, revision: Revision) -> Result<(), Fault> {
        self.ready=None;self.graph_revision=None;
        if inputs.layout.records!=joints.records || revision.joints!=joints.revision {
            return Err(refusal("frozen joint view/revision does not match reduced model inputs"));
        }
        if inputs.source_rows>u32::MAX/3 || !inputs.dt.is_finite() || inputs.dt<=0.0 {
            return Err(refusal("invalid frozen full-source rows/dt"));
        }
        let mut adapter=Adapter{device,storage:&mut self.storage,inputs,joints,revision,phase:"allocate"};
        self.core.rebuild(&mut adapter,inputs.layout,revision).map_err(|e| match e {
            model::Error::Invalid(s)=>refusal(s),model::Error::Device(f)=>f,
        })?;
        let domains=self.core.domains();
        self.storage.stats.domains=domains;
        (self.storage.stats.partition_reused,self.storage.stats.partition_ms)=self.core.partition_stats();
        // These are explicit regularization metadata, never B or local blocks.
        if inputs.layout.bodies!=0 {
            self.storage.body_shift.download(device)?;
            for &x in &self.storage.body_shift.host()[..inputs.layout.bodies as usize] {
                if !x.is_finite() || x<0.0 {return Err(refusal("invalid successful body shift metadata"));}
                self.storage.stats.body_shifted+=(x!=0.0) as u32;
                self.storage.stats.body_shift_max=self.storage.stats.body_shift_max.max(x);
            }
        }
        if domains!=0 {
            self.storage.shift.download(device)?;
            for &x in &self.storage.shift.host()[..domains as usize] {
                if !x.is_finite() || x<0.0 || x>1.001e-4 {return Err(refusal("invalid successful local jitter metadata"));}
                self.storage.stats.local_jittered+=(x!=0.0) as u32;
                self.storage.stats.local_jitter_max=self.storage.stats.local_jitter_max.max(x);
            }
        }
        if super::constraint_space::diagnostic_selected(revision)?{self.constraint_diagnostic(device,inputs.constraints,domains)?;}
        self.ready=Some(revision);self.graph_revision=Some(revision);
        Ok(())
    }

    /// Encode into the caller's existing resident PCG graph/batch. The residual
    /// must already be projected; caller MUST retain its original projector
    /// immediately around every apply. This method never overwrites residual.
    /// Result and residual must be disjoint and each hold 3*model.rows scalars.
    pub unsafe fn encode<E: Encoder + ?Sized>(&mut self, encoder: &mut E, residual: Handle,
        result: Handle, current: Revision) -> Result<(), Fault> {
        if self.ready!=Some(current) || !self.core.is_current(current) {
            return Err(refusal("stale/unready reduced model factors"));
        }
        let domains=self.core.domains();
        if domains==0 {return Ok(());}
        let args=RsmApplyArgs {member_offset:self.storage.member_offset.handle(),
            members:self.storage.members.handle(),inverse:self.storage.inverse.handle(),
            residual,result,apply_fault:self.storage.fault.handle(),receipt:self.storage.receipt.handle(),
            extent:domains,count:domains,seam_arena_count:0};
        encoder.elements(&args,domains)
    }

    /// Small opt-in frozen diagnostic only; no production numerical readback.
    pub fn diagnostic_seed_handles(&self)->(Handle,Handle){
        (self.storage.body_seed.handle(),self.storage.body_inertia.handle())
    }
    /// Diagnostic-only original duplicate CSR, before any signed coalescing.
    /// The separate same-revision old->unique comparison must already pass.
    /// Production gather/ModelView remain UNIQUE; no numerical buffers change.
    pub fn diagnostic_raw_model_view(&mut self)->Option<ModelView>{
        let revision=self.ready?;
        if !self.core.is_current(revision)||self.storage.coalescer.diagnostic_revision()!=Some(revision){return None;}
        Some(ModelView{rows:self.storage.stats.rows,blocks:self.storage.stats.raw_blocks,
            offset:self.storage.offset.handle(),column:self.storage.column.handle(),
            value:self.storage.value.handle(),revision})
    }
    pub fn stats(&self) -> Stats {self.storage.stats}
    /// Current disjoint domain membership, used only to omit a duplicate
    /// joint-edge correction already contained within one local factor.
    pub unsafe fn domain_owners<D:Device>(&self,device:&mut D)->Result<Vec<u32>,Fault> {
        let revision=self.ready.ok_or_else(||refusal("domain owners unavailable"))?;
        if !self.core.is_current(revision){return Err(refusal("stale domain owners"));}
        super::constraint_space::read_u32(device,self.storage.assignment.handle(),self.storage.stats.rows as usize)
    }
    /// Structural IDs already downloaded by current frozen model rebuild.
    /// CPU cache captured by graph() from exact used device intervals. Naming
    /// a ReadbackBuffer in a later read-only dispatch invalidates host(); never
    /// access those readback banks here. Cache is published only after rebuild.
    pub fn graph_snapshot(&self)->Option<(Vec<u32>,Vec<u32>)> {
        let revision=self.ready?;
        if self.graph_revision!=Some(revision) || !self.core.is_current(revision) {
            return None;
        }
        let s=&self.storage;
        if s.graph_offset.len()!=s.stats.rows as usize+1 ||
            s.graph_column.len()!=s.stats.blocks as usize ||
            s.graph_offset.last().copied()!=Some(s.stats.blocks) {return None;}
        Some((s.graph_offset.clone(),s.graph_column.clone()))
    }


    pub fn graph_weights(&self)->Option<Vec<u32>>{
        let r=self.ready?;if self.graph_revision!=Some(r)||!self.core.is_current(r){return None;}
        Some(self.storage.graph_weight.clone())
    }

    pub fn check<D:Device>(&mut self, device:&mut D)->Result<ApplyReceipt,Fault> {
        if self.ready.is_none(){return Err(refusal("checking unready reduced model"));}
        let domains=self.core.domains();
        // Success observation is exactly16bytes, independent of model size.
        // Fault array was zero after factor and apply only adds sticky faults.
        self.storage.receipt.download(device)?;
        let r=self.storage.receipt.host();
        let receipt=ApplyReceipt{dispatches:r[0],domain_successes:r[1],domain_faults:r[2]};
        if r[3]!=0{return Err(refusal("device apply receipt counter overflow"));}
        if receipt.domain_faults!=0 {
            self.storage.fault.download(device)?;
            if let Some((i,&code))=self.storage.fault.host()[..domains as usize].iter().enumerate().find(|(_,c)|**c!=0){
                return Err(refusal(format!("apply domain {i}: sticky device fault {code}")));
            }
            return Err(refusal("device apply fault receipt is nonzero"));
        }
        let expected=u64::from(domains)*u64::from(receipt.dispatches);
        if expected>u64::from(u32::MAX) || expected!=u64::from(receipt.domain_successes) {
            return Err(refusal("device apply receipt/domain coverage mismatch"));
        }
        Ok(receipt)
    }
    /// Future hierarchy must consume this exact CSR, including additive duplicates.
    pub fn model_view(&mut self) -> Option<ModelView> {
        let revision=self.ready?;
        if self.storage.stats.rows==0{return Some(ModelView{rows:0,blocks:0,offset:self.storage.offset.handle(),
            column:self.storage.column.handle(),value:self.storage.value.handle(),revision});}
        let csr=self.storage.coalescer.view(revision)?;
        Some(ModelView{rows:self.storage.stats.rows,blocks:csr.blocks,offset:csr.offset,column:csr.column,value:csr.value,revision})
    }
}

struct Adapter<'a,D:Device> {device:&'a mut D,storage:&'a mut Storage,inputs:Inputs,joints:FrozenJointBlockView,revision:Revision,phase:&'static str}

impl<D:Device> Backend for Adapter<'_,D> {
    type Error=Fault;
    fn allocate(&mut self,rows:u32,capacity:usize,records:u32)->Result<(),Fault>{
        let s=&mut self.storage;let d=&mut *self.device;
        s.stats=Stats{rows,..Stats::default()};s.capacity=capacity;
        s.graph_offset.clear();s.graph_column.clear();s.graph_weight.clear();
        // Empty model skips graph(), but its exact CSR structure is [0].
        if rows==0{s.graph_offset.push(0);}
        s.offset.size(d,rows as usize+1,AllocLabel("rsm.offset"))?;
        s.cursor.size(d,(rows as usize).max(1),AllocLabel("rsm.cursor"))?;
        s.base_count.size(d,(rows as usize).max(1),AllocLabel("rsm.base_count"))?;
        s.projection_fault.size(d,1,AllocLabel("rsm.projection_fault"))?;
        s.column.size(d,capacity.max(1),AllocLabel("rsm.column"))?;
        s.value.size(d,capacity.max(1)*9,AllocLabel("rsm.value"))?;
        s.body_seed.size(d,(self.inputs.layout.bodies as usize*36).max(1),AllocLabel("rsm.body_seed"))?;
        s.body_inertia.size(d,(self.inputs.layout.bodies as usize*36).max(1),AllocLabel("rsm.private_analytic_inertia"))?;
        if self.inputs.layout.bodies==0 && s.inactive_body_state.len()==0 {
            s.inactive_body_state.size(d,1,AllocLabel("rsm.inactive_body_state"))?;
            d.fill_zero(s.inactive_body_state.handle(),std::mem::size_of::<crate::data::PdrdRigidState>())?;
        }
        s.body_shift.size(d,(self.inputs.layout.bodies as usize).max(1),AllocLabel("rsm.body_shift"))?;
        let faults=rows.max(records).max(1);
        s.fault.size(d,faults as usize,AllocLabel("rsm.fault"))?;
        s.receipt.size(d,4,AllocLabel("rsm.apply_receipt"))?;
        // Only exact used CSR intervals are ever downloaded or read. Capacity
        // tails need no initialization and never contribute to B or its graph.
        for (array,count,value) in [(s.offset.handle(),rows+1,0),
            (s.projection_fault.handle(),1,0),(s.fault.handle(),faults,0),
            (s.receipt.handle(),4,0)] {
            let args=VecFillU32Args{array,value,count,seam_arena_count:0};
            unsafe{d.launch("rsm.initialize_metadata",&args,count)?;}
        }
        Ok(())
    }
    fn body_seed(&mut self,bodies:u32)->Result<(),Fault>{
        self.phase="body_seed";
        let a=RsmBodySeedArgs{raw_body:self.inputs.raw_body,body_seed:self.storage.body_seed.handle(),
            body_shift:self.storage.body_shift.handle(),fault:self.storage.fault.handle(),
            extent:bodies,count:bodies,seam_arena_count:0};
        unsafe{self.device.launch("rsm.body_seed",&a,bodies)?;}
        // EXACT original analytic inertia; subtract this when full C is
        // projected, rather than guessing a per-vertex mass/drag decomposition.
        let inertia=PdrdAssembleInertiaRowArgs{state:self.inputs.body_state,
            blocks:self.storage.body_inertia.handle(),dt:self.inputs.dt,count:bodies,seam_arena_count:0};
        unsafe{self.device.launch("rsm.private_analytic_inertia",&inertia,bodies).map(|_|())}
    }
    fn base_count(&mut self,layout:Layout)->Result<(),Fault>{
        let r=self.inputs.rows;let n=layout.cloth+2*layout.bodies;
        let threads=n.max(self.inputs.source_rows);
        let a=RsmBaseCountArgs{vertex_body:self.inputs.vertex_body,vertex_to_free:self.inputs.vertex_to_free,
            dynamic_index:r.dynamic_index,dynamic_offset:r.dynamic_offset,reference_index:r.reference_index,
            reference_offset:r.reference_offset,fixed_index:r.fixed_index,fixed_offset:r.fixed_offset,
            transpose_pair:r.transpose_pair,transpose_offset:r.transpose_offset,row_count:self.storage.offset.handle(),
            projection_fault:self.storage.projection_fault.handle(),n_cloth:layout.cloth,bodies:layout.bodies,
            extent:self.inputs.source_rows,count:threads,seam_arena_count:0};
        unsafe{self.device.launch("rsm.base_count",&a,threads)?;}
        // Preserve each row's BASE budget before joint_count adds its own.
        self.device.copy(self.storage.base_count.handle(),0,self.storage.offset.handle(),0,n as usize*4)
    }
    fn joint_count(&mut self,records:u32,rows:u32)->Result<(),Fault>{
        self.phase="joint_count";
        let a=RsmJointCountArgs{scalar_ids:self.joints.scalar_ids,scalar_count:self.joints.scalar_count,
            row_count:self.storage.offset.handle(),fault:self.storage.fault.handle(),nrows:rows,
            extent:records,count:records,seam_arena_count:0};
        unsafe{self.device.launch("rsm.joint_count",&a,records).map(|_|())}
    }
    fn check_fault(&mut self,extent:u32)->Result<(),Fault>{
        self.storage.fault.download(self.device)?;
        if let Some((i,&code))=self.storage.fault.host()[..extent as usize].iter().enumerate().find(|(_,c)|**c!=0){
            return Err(refusal(format!("device phase {} fault at {i}: code {code}",self.phase)));
        }
        Ok(())
    }
    fn scan_counts(&mut self,rows:u32)->Result<(),Fault>{unsafe{
        self.storage.scan.exclusive(self.device,"rsm.scan_counts",self.storage.offset.handle(),rows)?;
        // Refuse before either fill kernel can write outside column/value.
        // Read only the final u32 offset; numerical B stays on the device.
        let byte_offset=(rows as usize).checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(||refusal("CSR total byte offset overflows usize"))?;
        let mut raw=[0u8;4];
        self.device.read(self.storage.offset.handle(),byte_offset,&mut raw)?;
        let total=u32::from_le_bytes(raw) as usize;
        self.storage.stats.raw_blocks=total as u32;
        if total>self.storage.capacity {
            return Err(refusal(format!("CSR scan total {total} exceeds checked capacity {} before fill",self.storage.capacity)));
        }
        Ok(())
    }}
    fn base_fill(&mut self,layout:Layout)->Result<(),Fault>{
        let r=self.inputs.rows;let n=layout.cloth+2*layout.bodies;
        let threads=n.max(self.inputs.source_rows);
        // All base and joint writers reserve from the original scanned start.
        self.device.copy(self.storage.cursor.handle(),0,self.storage.offset.handle(),0,n as usize*4)?;
        let body_state=if layout.bodies!=0{self.inputs.body_state}else{self.storage.inactive_body_state.handle()};
        let a=RsmBaseFillArgs{row_to_vertex:self.inputs.row_to_vertex,vertex_body:self.inputs.vertex_body,
            vertex_to_free:self.inputs.vertex_to_free,rotated_rest:self.inputs.rotated_rest,body_state,
            dynamic_index:r.dynamic_index,dynamic_value:r.dynamic_value,dynamic_offset:r.dynamic_offset,
            reference_index:r.reference_index,reference_value:r.reference_value,reference_offset:r.reference_offset,
            global_value:r.global_value,fixed_index:r.fixed_index,fixed_offset:r.fixed_offset,fixed_value:r.fixed_value,
            transpose_pair:r.transpose_pair,transpose_offset:r.transpose_offset,diagonal:r.diagonal,
            body_seed:self.storage.body_seed.handle(),body_inertia:self.storage.body_inertia.handle(),
            offset:self.storage.offset.handle(),base_count:self.storage.base_count.handle(),cursor:self.storage.cursor.handle(),
            column:self.storage.column.handle(),value:self.storage.value.handle(),n_cloth:layout.cloth,
            projection_fault:self.storage.projection_fault.handle(),bodies:layout.bodies,
            extent:self.inputs.source_rows,count:threads,seam_arena_count:0};
        unsafe{self.device.launch("rsm.base_fill",&a,threads)?;}
        // Sticky count/fill mapping, reservation or arithmetic faults are
        // private and checked before any joint fill/local numerical factor.
        self.storage.projection_fault.download(self.device)?;
        if self.storage.projection_fault.host()[0]!=0 {
            return Err(refusal(format!("full operator projection refused: {} device faults",self.storage.projection_fault.host()[0])));
        }
        Ok(())
    }
    fn joint_fill(&mut self,records:u32)->Result<(),Fault>{
        let a=RsmJointFillArgs{scalar_ids:self.joints.scalar_ids,scalar_count:self.joints.scalar_count,
            local_hessian:self.joints.local_hessian,cursor:self.storage.cursor.handle(),
            column:self.storage.column.handle(),value:self.storage.value.handle(),extent:records,count:records,seam_arena_count:0};
        unsafe{self.device.launch("rsm.joint_fill",&a,records).map(|_|())}
    }
    fn validate_csr(&mut self,rows:u32)->Result<(),Fault>{
        self.phase="csr_check";
        let a=RsmCheckArgs{offset:self.storage.offset.handle(),cursor:self.storage.cursor.handle(),
            column:self.storage.column.handle(),value:self.storage.value.handle(),fault:self.storage.fault.handle(),
            extent:rows,count:rows,seam_arena_count:0};
        unsafe{self.device.launch("rsm.check",&a,rows).map(|_|())}
    }
    fn graph(&mut self,rows:u32)->Result<(Vec<u32>,Vec<u32>),Fault>{
        let raw_blocks=self.storage.stats.raw_blocks;
        unsafe{self.storage.coalescer.rebuild(self.device,self.storage.offset.handle(),self.storage.column.handle(),
            self.storage.value.handle(),self.inputs.vertex_body,self.inputs.vertex_to_free,rows,raw_blocks,
            self.inputs.source_rows,self.inputs.layout.cloth,self.inputs.layout.bodies,self.revision)?;}
        let (offset,column,weights)=self.storage.coalescer.graph();
        self.storage.graph_offset=offset.to_vec();self.storage.graph_column=column.to_vec();self.storage.graph_weight=weights.to_vec();
        let st=self.storage.coalescer.stats();
        self.storage.stats.blocks=st.unique_blocks;self.storage.stats.coalesce_cache_hit=st.topology_cache_hit;
        self.storage.stats.coalesce_topology_ms=st.topology_ms;self.storage.stats.coalesce_numeric_ms=st.numeric_ms;
        self.storage.stats.coalesce_total_ms=st.total_ms;
        self.storage.stats.edge_weight_ms=st.edge_weight_ms;
        Ok((self.storage.graph_offset.clone(),self.storage.graph_column.clone()))
    }
    fn graph_weights(&self)->Option<&[u32]>{Some(&self.storage.graph_weight)}
    fn upload_domains(&mut self,domains:&Domains)->Result<(),Fault>{
        let s=&mut self.storage;let d=&mut *self.device;
        let n=domains.offset.len()-1;let cells=n.checked_mul(model::DENSE_STRIDE).ok_or_else(||refusal("dense extent overflow"))?;
        s.member_offset.size(d,domains.offset.len(),AllocLabel("rsm.member_offset"))?;
        s.member_offset.write(d,0,&domains.offset)?;
        s.members.size(d,domains.members.len().max(1),AllocLabel("rsm.members"))?;
        s.members.write(d,0,&domains.members)?;
        s.assignment.size(d,domains.assignment.len().max(1),AllocLabel("rsm.assignment"))?;
        s.assignment.write(d,0,&domains.assignment)?;
        s.dense.size(d,cells.max(1),AllocLabel("rsm.dense"))?;
        s.normalized.size(d,cells.max(1),AllocLabel("rsm.normalized"))?;
        s.lower.size(d,cells.max(1),AllocLabel("rsm.lower"))?;
        s.inverse.size(d,cells.max(1),AllocLabel("rsm.inverse"))?;
        s.scale.size(d,(48*n).max(1),AllocLabel("rsm.scale"))?;
        s.shift.size(d,n.max(1),AllocLabel("rsm.shift"))?;
        Ok(())
    }
    fn gather(&mut self,domains:u32)->Result<(),Fault>{
        self.phase="gather";
        let s=&mut self.storage;
        let csr=s.coalescer.view(self.revision).ok_or_else(||refusal("unready coalesced CSR at gather"))?;
        let a=RsmGatherArgs{offset:csr.offset,column:csr.column,value:csr.value,
            member_offset:s.member_offset.handle(),members:s.members.handle(),dense:s.dense.handle(),fault:s.fault.handle(),
            extent:domains,count:domains,seam_arena_count:0};
        unsafe{self.device.launch("rsm.gather",&a,domains)?;}
        let q=self.inputs.constraints;
        if q.revision!=self.revision{return Err(refusal("stale frozen constraint Q"));}
        let a=RqcLocalArgs{member_offset:s.member_offset.handle(),members:s.members.handle(),q:q.q,
            dense:s.dense.handle(),fault:s.fault.handle(),cloth:q.cloth,extent:domains,count:domains,seam_arena_count:0};
        unsafe{self.device.launch("rqc.local",&a,domains).map(|_|())}
    }
    fn factor(&mut self,domains:u32)->Result<(),Fault>{
        self.phase="factor";
        let s=&mut self.storage;
        let a=RsmFactorArgs{member_offset:s.member_offset.handle(),dense:s.dense.handle(),normalized:s.normalized.handle(),
            lower:s.lower.handle(),inverse:s.inverse.handle(),scale:s.scale.handle(),shift:s.shift.handle(),fault:s.fault.handle(),
            extent:domains,count:domains,seam_arena_count:0};
        unsafe{self.device.launch("rsm.factor",&a,domains).map(|_|())}
    }
    // Rebuild adapter deliberately has no residual handles/projector. Solver
    // applies via encode() surrounded by its authoritative existing projector.
    fn project_input(&mut self)->Result<(),Fault>{Err(refusal("use caller-projected encode API"))}
    fn apply(&mut self,_:u32)->Result<(),Fault>{Err(refusal("use caller-projected encode API"))}
    fn project_output(&mut self)->Result<(),Fault>{Err(refusal("use caller-projected encode API"))}
}

impl ReducedModel{
 unsafe fn constraint_diagnostic<D:Device>(&mut self,d:&mut D,qv:super::constraint_space::View,domains:u32)->Result<(),Fault>{
 use super::constraint_space::{read_f32,read_u32,q_entry};
 if domains>256{return Err(refusal("opt-in local constraint diagnostic bound 256 domains"));}
 let s=&mut self.storage;let off=read_u32(d,s.member_offset.handle(),domains as usize+1)?;
 let members=read_u32(d,s.members.handle(),*off.last().unwrap()as usize)?;
 let q=read_f32(d,qv.q,36*(qv.cloth+qv.bodies)as usize)?;
 let dense=read_f32(d,s.dense.handle(),2304*domains as usize)?;
 let inv=read_f32(d,s.inverse.handle(),2304*domains as usize)?;
 let mut comm=0f64;let mut inv_error=0f64;let mut all_finite=true;
 for b in 0..domains as usize{let rows=&members[off[b]as usize..off[b+1]as usize];let n=3*rows.len();
 let qe=|i:usize,j:usize|q_entry(&q,qv.cloth,rows[i/3],i%3,rows[j/3],j%3);
 let a=&dense[2304*b..2304*b+n*n];let g=&inv[2304*b..2304*b+n*n];
 let scale=a.iter().map(|&x|f64::from(x).abs()).fold(0f64,f64::max);
 for i in 0..n{for j in 0..n{let left=(0..n).map(|k|qe(i,k)*f64::from(a[k*n+j])).sum::<f64>();let right=(0..n).map(|k|f64::from(a[i*n+k])*qe(k,j)).sum::<f64>();let error=(left-right).abs();all_finite&=error.is_finite();if scale>0.{comm=comm.max(error/scale);}}}
 let v:Vec<f64>=(0..n).map(|i|(((7*i)%11)as f64-5.)/8.).collect();
 let r:Vec<f64>=(0..n).map(|i|(0..n).map(|j|qe(i,j)*v[j]).sum()).collect();
 let y:Vec<f64>=(0..n).map(|i|(0..=i).map(|j|f64::from(g[i*n+j])*r[j]).sum()).collect();
 let z:Vec<f64>=(0..n).map(|i|(i..n).map(|j|f64::from(g[j*n+i])*y[j]).sum()).collect();
 let az:Vec<f64>=(0..n).map(|i|(0..n).map(|j|f64::from(a[i*n+j])*z[j]).sum::<f64>()-r[i]).collect();
 let den=r.iter().map(|x|x.abs()).sum::<f64>();let err=(0..n).map(|i|(0..n).map(|j|qe(i,j)*az[j]).sum::<f64>().abs()).sum::<f64>();if den>0.{inv_error=inv_error.max(err/den);}all_finite&=err.is_finite();
 }
 let bound=4096.*f64::from(f32::EPSILON);if !all_finite||comm>bound{return Err(refusal(format!("local constraint commutator invariant {comm} bound {bound}")));}
 eprintln!("SCHWARZ_CONSTRAINT_LOCAL_CHECK {}",serde_json::json!({"revision":qv.revision.operator,"domains":domains,"Q_Dhat_commutator_relative_max":comm,"commutator_bound":bound,"projected_local_inverse_residual_max":inv_error,"inverse_residual_scope":"FP64 probe of actual stored original factors; original normalized jitter retained, reported not a convergence relaxation"}));Ok(())
 }
}
