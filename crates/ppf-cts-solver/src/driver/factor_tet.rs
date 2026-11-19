//! Frozen physical TET operator; full CSR survives as roundoff approximation
//! for preconditioning/reference only. Never subtract assembled products.
use ppf_cts_compute::{AllocLabel,Buffer,ReadbackBuffer,Device,Encoder,EncoderExt,Fault,Handle};
use super::{kernels::*,fixedcsr::FixedCsr,scene::{Fatal,FatalResult}};
fn refusal(detail:impl Into<String>)->Fault {Fault::Platform{call:"frozen physical tet factor",detail:detail.into()}}
#[derive(Clone,Copy)]
pub struct View {
    pub non_tet:Handle,
    tet:Handle,factor:Handle,weight:Handle,response:Handle,
    adj_offset:Handle,adj:Handle,mask:Handle,fault:Handle,
    pub rows:u32,tets:u32,
}
#[derive(Clone,Copy)]
pub struct CaptureSnapshot {
    pub non_tet:Handle,pub tet:Handle,pub factor:Handle,pub weight:Handle,
    pub mask:Handle,pub fault:Handle,pub rows:u32,pub tets:u32,
}
impl View {
    /// Read-only addresses of the actual frozen producer, no numerical rebuild.
    pub fn capture_snapshot(&self)->CaptureSnapshot {
        CaptureSnapshot{non_tet:self.non_tet,tet:self.tet,factor:self.factor,
            weight:self.weight,mask:self.mask,fault:self.fault,rows:self.rows,tets:self.tets}
    }
    /// Optional diagnostic reuses exact frozen coefficients/adjacency, replacing
    /// only scratch response and sticky fault destinations with private buffers.
    pub fn with_capture_scratch(&self,response:Handle,fault:Handle)->Self {
        Self{response,fault,..*self}
    }

    pub unsafe fn encode<E:Encoder+?Sized>(&self,e:&mut E,input:Handle,result:Handle,
        curvature:Handle,absolute:Handle)->Result<(),Fault> {
        self.encode_kind(e,input,input,result,curvature,absolute,0)
    }
    /// Append the exact force action without unused per-row metrics.
    pub unsafe fn encode_force_only<E: Encoder + ?Sized>(
        &self, encoder: &mut E, input: Handle, result: Handle,
    ) -> Result<(), Fault> {
        let evaluate = FttEvaluateArgs {
            tet: self.tet, factor: self.factor, weight: self.weight,
            input, target: input, mask: self.mask, response: self.response,
            fault: self.fault, lift: 0, extent: self.tets, count: self.tets,
            seam_arena_count: 0,
        };
        encoder.elements(&evaluate, self.tets)?;
        let gather = FttGatherForceOnlyArgs {
            adj_offset: self.adj_offset, adj: self.adj, mask: self.mask,
            response: self.response, result, fault: self.fault,
            extent: self.rows, count: self.rows, seam_arena_count: 0,
        };
        encoder.elements(&gather, self.rows)
    }
    pub(super) unsafe fn encode_batch8<E:Encoder+?Sized>(
        &self, e:&mut E, input:Handle, result:Handle, curvature:Handle,
        absolute:Handle, status:Handle,
    ) -> Result<(),Fault> {
        let eval = ResidentGuardFttEvaluateArgs {
            tet:self.tet, factor:self.factor, weight:self.weight, input, target:input,
            mask:self.mask, response:self.response, fault:self.fault, lift:0,
            status, extent:self.tets, count:self.tets, seam_arena_count:0,
        };
        e.elements(&eval,self.tets)?;
        let gather = ResidentGuardFttGatherArgs {
            adj_offset:self.adj_offset, adj:self.adj, mask:self.mask, tet:self.tet,
            response:self.response, result, curvature, absolute, fault:self.fault, lift:0,
            status, extent:self.rows, count:self.rows, seam_arena_count:0,
        };
        e.elements(&gather,self.rows)
    }
    unsafe fn encode_kind<E:Encoder+?Sized>(&self,e:&mut E,input:Handle,target:Handle,result:Handle,
        curvature:Handle,absolute:Handle,lift:u32)->Result<(),Fault> {
        let a=FttEvaluateArgs{tet:self.tet,factor:self.factor,weight:self.weight,input,target,
            mask:self.mask,response:self.response,fault:self.fault,lift,
            extent:self.tets,count:self.tets,seam_arena_count:0};
        e.elements(&a,self.tets)?;
        let a=FttGatherArgs{adj_offset:self.adj_offset,adj:self.adj,mask:self.mask,tet:self.tet,
            response:self.response,result,curvature,absolute,fault:self.fault,lift,
            extent:self.rows,count:self.rows,seam_arena_count:0};
        e.elements(&a,self.rows)
    }
}
#[derive(Default)]
pub struct FactorTet {
    pub non_tet:ReadbackBuffer<f32>,tet_only:ReadbackBuffer<f32>,
    factor:Buffer<f32>,weight:Buffer<f32>,response:Buffer<f32>,
    adj_offset:Buffer<u32>,adj:Buffer<u32>,fault:ReadbackBuffer<u32>,
    dummy_scalar:Buffer<f32>,rows:u32,tets:u32,ready:bool,
}
impl FactorTet {
    pub unsafe fn initialize<D:Device>(&mut self,d:&mut D,data:&crate::data::DataSet,
        rows:usize,floats:usize)->Result<(),Fault> {
        let mesh: &[crate::data::Vec4u]=super::scene::slice(&data.mesh.mesh.tet);
        if rows>u32::MAX as usize || mesh.len()>u32::MAX as usize/81 || floats>u32::MAX as usize {
            return Err(refusal("factor cache index extent overflow"));
        }
        self.rows=rows as u32;self.tets=mesh.len() as u32;self.ready=false;
        if mesh.is_empty(){return Ok(());}
        let mut per_row=vec![Vec::<u32>::new();rows];
        for (element,tet) in mesh.iter().enumerate() {for slot in 0..4 {
            let v=tet[slot] as usize;
            if v>=rows{return Err(refusal("factor tet vertex outside current full rows"));}
            per_row[v].push(4*element as u32+slot as u32);
        }}
        let mut offset=vec![0u32];let mut adj=Vec::with_capacity(4*mesh.len());
        for neighbors in per_row {adj.extend(neighbors);offset.push(adj.len() as u32);}
        self.adj_offset.size(d,offset.len(),AllocLabel("ftt.adj_offset"))?;
        self.adj_offset.write(d,0,&offset)?;
        self.adj.size(d,adj.len(),AllocLabel("ftt.adj"))?;self.adj.write(d,0,&adj)?;
        self.non_tet.size(d,floats.max(1),AllocLabel("ftt.non_tet"))?;
        self.tet_only.size(d,floats.max(1),AllocLabel("ftt.tet_only"))?;
        self.factor.size(d,81*mesh.len(),AllocLabel("ftt.factor"))?;
        self.weight.size(d,9*mesh.len(),AllocLabel("ftt.weight"))?;
        self.response.size(d,11*mesh.len(),AllocLabel("ftt.response"))?;
        self.fault.size(d,rows.max(mesh.len()).max(1),AllocLabel("ftt.fault"))?;
        self.dummy_scalar.size(d,rows.max(1),AllocLabel("ftt.lift_scalar"))?;
        Ok(())
    }
    pub fn active(&self)->bool {self.tets!=0}
    pub unsafe fn begin_tet<D:Device>(&mut self,d:&mut D,fixed:&mut FixedCsr<'_>)->FatalResult<()> {
        self.ready=false;
        if !self.active(){return Ok(());}
        self.non_tet.copy_from(d,fixed.value_readback())?;
        let a=VecFillU32Args{array:self.fault.handle(),value:0,
            count:self.rows.max(self.tets).max(1),seam_arena_count:0};
        d.launch("ftt.zero_fault",&a,a.count)?;
        fixed.clear(d)
    }
    /// Called immediately after tet-only embed. Restore pre-tet non-tet values
    /// so all later non-tet layers accumulate directly, without cancellation.
    pub fn end_tet<D:Device>(&mut self,d:&mut D,fixed:&mut FixedCsr<'_>)->FatalResult<()> {
        if !self.active(){return Ok(());}
        self.check(d)?;
        self.tet_only.copy_from(d,fixed.value_readback())?;
        fixed.copy_values_from(d,&self.non_tet)?;self.ready=true;Ok(())
    }
    pub fn cache_handles(&mut self)->(Handle,Handle,Handle) {
        (self.factor.handle(),self.weight.handle(),self.fault.handle())
    }
    /// Independent full reference = non-tet + tet-only, before contact reads it.
    pub unsafe fn full_reference<D:Device>(&mut self,d:&mut D,non_tet:Handle,
        full:Handle,floats:u32)->Result<(),Fault> {
        if !self.active(){return Ok(());}
        let a=FttCombineArgs{non_tet,tet:self.tet_only.handle(),full,extent:floats,count:floats,seam_arena_count:0};
        d.launch("ftt.full_reference",&a,floats)?;Ok(())
    }
    pub fn view(&mut self,tet:Handle,mask:Handle)->Option<View> {
        (self.active()&&self.ready).then(||View{non_tet:self.non_tet.handle(),tet,
            factor:self.factor.handle(),weight:self.weight.handle(),response:self.response.handle(),
            adj_offset:self.adj_offset.handle(),adj:self.adj.handle(),mask,fault:self.fault.handle(),
            rows:self.rows,tets:self.tets})
    }
    pub unsafe fn lift<D:Device>(&mut self,d:&mut D,tet:Handle,mask:Handle,
        eval_x:Handle,target:Handle,force:Handle)->Result<(),Fault> {
        let view=match self.view(tet,mask){Some(v)=>v,None=>return Ok(())};
        let dummy=self.dummy_scalar.handle();
        d.run("ftt.dirichlet_lift",|e|view.encode_kind(e,eval_x,target,force,dummy,dummy,1))?;
        Ok(())
    }
    /// AFTER original non-tet+dynamic lift/prescribe. Save the actual masked
    /// non-tet operator, mask tet-only approximation without lifting again, and
    /// reconstruct full approximation for original preconditioner/model CSR.
    pub unsafe fn finish<D:Device>(&mut self,d:&mut D,fixed:&mut FixedCsr<'_>,
        mask:Handle,offset:Handle,column:Handle)->FatalResult<()> {
        if !self.active(){return Ok(());}
        self.non_tet.copy_from(d,fixed.value_readback())?;
        let a=FttMaskValuesArgs{offset,column,mask,value:self.tet_only.handle(),
            extent:self.rows,count:self.rows,seam_arena_count:0};
        d.launch("ftt.mask_tet_approximation",&a,self.rows)?;
        let count=self.non_tet.len() as u32;
        let a=FttCombineArgs{non_tet:self.non_tet.handle(),tet:self.tet_only.handle(),full:fixed.values_mut(),
            extent:count,count,seam_arena_count:0};
        d.launch("ftt.full_preconditioner",&a,count)?;self.check(d)?;Ok(())
    }
    pub fn check<D:Device>(&mut self,d:&mut D)->Result<(),Fault> {
        if !self.active(){return Ok(());}
        self.fault.download(d)?;
        if let Some((i,c))=self.fault.host().iter().enumerate().find(|(_,c)|**c!=0) {
            return Err(refusal(format!("sticky factor fault {c} at {i}")));
        }
        Ok(())
    }
}
