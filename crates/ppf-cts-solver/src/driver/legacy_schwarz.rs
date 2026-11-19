//! Full frozen legacy joint blocks, appended to the material export on device.
use ppf_cts_compute::{AllocLabel,Buffer,Device,Fault,Handle,Pod,ReadbackBuffer};
use super::{joints,kernels::LegacySchwarzExportArgs,reduced_schwarz::FrozenJointBlockView};
use crate::{joint_gpu::BallAnchorGpu,joint_angular_gpu::AngularJointGpuV2,
    joint_angular_control_gpu::AngularControlGpu,joint_angular_extension_gpu as ext,
    restoring_curve_gpu::{CurveDescriptor,CurveNode,CurveBinding,CurveHistory}};

fn refusal(detail:impl Into<String>)->Fault{Fault::Platform{call:"legacy reduced Schwarz frozen export",detail:detail.into()}}
fn dummy<T:Pod,D:Device>(buffer:&mut Buffer<T>,device:&mut D)->Result<Handle,Fault>{
    if buffer.len()==0{buffer.size(device,1,AllocLabel("legacy_schwarz.inactive_typed"))?;
        device.fill_zero(buffer.handle(),std::mem::size_of::<T>())?;}
    Ok(buffer.handle())
}

#[derive(Default)]
pub struct LegacyJointBlocks {
    ids:Buffer<u32>,hessian:Buffer<f32>,count:Buffer<u32>,fault:ReadbackBuffer<u32>,revision:u64,
    ball:Buffer<BallAnchorGpu>,angular:Buffer<AngularJointGpuV2>,control:Buffer<AngularControlGpu>,
    limits:Buffer<ext::AngularSoftLimitGpuV1>,drives:Buffer<ext::AngularDriveGpuV1>,keys:Buffer<ext::AngularDriveDeviceKeyV1>,
    descriptors:Buffer<CurveDescriptor>,nodes:Buffer<CurveNode>,bindings:Buffer<CurveBinding>,history:Buffer<CurveHistory>,
}

impl LegacyJointBlocks {
    pub(super) fn release(&mut self,d:&mut impl Device)->Result<(),Fault>{
        let mut first=None;macro_rules! release {($x:expr)=>{if let Err(e)=$x.free(d){if first.is_none(){first=Some(e);}}};}
        release!(self.ids);release!(self.hessian);release!(self.count);release!(self.fault);release!(self.ball);release!(self.angular);release!(self.control);release!(self.limits);release!(self.drives);release!(self.keys);release!(self.descriptors);release!(self.nodes);release!(self.bindings);release!(self.history);
        match first {Some(e)=>Err(e),None=>Ok(())}
    }

    /// Every frozen Newton call refreshes from the same immutable View as A.
    /// Material view remains independent; neither export writes live fault,
    /// observations, history, state, controls or original global operator.
    pub unsafe fn refresh<D:Device>(&mut self,device:&mut D,source:joints::View,
        material:FrozenJointBlockView)->Result<FrozenJointBlockView,Fault>{
        let legacy=if source.count==0{0}else{
            source.count.checked_mul(2).and_then(|n|n.checked_add(source.control_count))
                .and_then(|n|n.checked_add(source.extension_count)).ok_or_else(||refusal("legacy record count overflow"))?};
        if legacy==0{return Ok(material);}
        if !source.dt.is_finite()||source.dt<=0.||!source.sample_time_s.is_finite(){return Err(refusal("invalid frozen dt/sample time"));}
        if source.extension_count>source.control_count{return Err(refusal("extension count exceeds control record storage"));}
        if source.base>source.dim||(source.dim-source.base)%6!=0||(source.dim-source.base)/6!=source.bodies{
            return Err(refusal("legacy reduced layout mismatch"));}
        if let Some(curves)=source.curves{if curves.dt.to_bits()!=source.dt.to_bits(){return Err(refusal("curve dt differs from frozen view"));}}
        let records=material.records.checked_add(legacy).ok_or_else(||refusal("combined record count overflow"))?;
        if records>u32::MAX/576{return Err(refusal("combined joint indexing exceeds u32"));}
        let n=records as usize;let start=material.records as usize;
        self.ids.size(device,n.checked_mul(24).ok_or_else(||refusal("ID extent overflow"))?,AllocLabel("legacy_schwarz.combined_ids"))?;
        self.hessian.size(device,n.checked_mul(576).ok_or_else(||refusal("Hessian extent overflow"))?,AllocLabel("legacy_schwarz.combined_hessian"))?;
        self.count.size(device,n,AllocLabel("legacy_schwarz.combined_count"))?;
        self.fault.size(device,legacy as usize,AllocLabel("legacy_schwarz.export_fault"))?;
        if start!=0{
            device.copy(self.ids.handle(),0,material.scalar_ids,0,start*24*4)?;
            device.copy(self.hessian.handle(),0,material.local_hessian,0,start*576*4)?;
            device.copy(self.count.handle(),0,material.scalar_count,0,start*4)?;
        }
        // Generated entries resolve even inactive pointers: use correctly typed
        // one-element storage, including CurveHistory's actual 8-byte alignment.
        let ball=if source.count!=0{source.ball}else{dummy(&mut self.ball,device)?};
        let angular=if source.count!=0{source.angular}else{dummy(&mut self.angular,device)?};
        let control=if source.control_count!=0{source.control}else{dummy(&mut self.control,device)?};
        let limits=if source.extension_count!=0{source.limits}else{dummy(&mut self.limits,device)?};
        let drives=if source.drive_count!=0{source.drives}else{dummy(&mut self.drives,device)?};
        let keys=if source.drive_count!=0{source.device_keys}else{dummy(&mut self.keys,device)?};
        let curves=source.curves;
        let descriptors=match curves{Some(c)=>c.descriptors,None=>dummy(&mut self.descriptors,device)?};
        let nodes=match curves{Some(c)=>c.nodes,None=>dummy(&mut self.nodes,device)?};
        let bindings=match curves{Some(c)=>c.bindings,None=>dummy(&mut self.bindings,device)?};
        let history=match curves{Some(c)=>c.history,None=>dummy(&mut self.history,device)?};
        let args=LegacySchwarzExportArgs{ball,angular,base_count:source.count,control,twist_context:source.twist_context,material_spin_context:source.material_spin_context,control_count:source.control_count,
            limits,extension_count:source.extension_count,drives,drive_count:source.drive_count,keys,
            state:source.state,reference:source.reference,bodies:source.bodies,base:source.base,dim:source.dim,
            dt:source.dt,sample_time_s:source.sample_time_s,curves_enabled:curves.is_some()as u32,
            descriptors,nodes,bindings,history,descriptor_count:curves.map_or(0,|c|c.descriptor_count),
            node_count:curves.map_or(0,|c|c.node_count),binding_count:curves.map_or(0,|c|c.binding_count),
            history_count:curves.map_or(0,|c|c.history_count),scalar_ids:self.ids.span(start*24,legacy as usize*24),
            local_hessian:self.hessian.span(start*576,legacy as usize*576),scalar_count:self.count.span(start,legacy as usize),
            fault:self.fault.handle(),extent:legacy,count:legacy,seam_arena_count:0};
        device.launch("legacy_schwarz.export",&args,legacy)?;
        self.fault.download(device)?;
        if let Some((record,code))=self.fault.host().iter().enumerate().find(|(_,c)|**c!=0){
            return Err(refusal(format!("frozen legacy record {record}: device fault {code}")));}
        self.revision=self.revision.checked_add(1).ok_or_else(||refusal("joint revision overflow"))?;
        Ok(FrozenJointBlockView{scalar_ids:self.ids.handle(),local_hessian:self.hessian.handle(),
            scalar_count:self.count.handle(),records,revision:self.revision})
    }
}

