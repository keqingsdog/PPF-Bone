//! Additive, frozen, topology-derived joint-edge preconditioner prototype.
//! It changes only the preconditioner and remains fixed for one PCG solve.
use std::collections::BTreeSet;
use ppf_cts_compute::{AllocLabel,Buffer,Device,Encoder,EncoderExt,Fault,Handle,ReadbackBuffer};
use super::{constraint_space, kernels::*, reduced_model::Revision,
    reduced_model_backend::ModelView, reduced_schwarz::FrozenJointBlockView};

fn refusal(s:impl Into<String>)->Fault {
    Fault::Platform { call:"joint patch preconditioner", detail:s.into() }
}

#[derive(Default)]
pub struct JointPatches {
    member_offset:Buffer<u32>, members:Buffer<u32>, weight:Buffer<f32>,
    incident_offset:Buffer<u32>, incident:Buffer<u32>, inverse:Buffer<f32>,
    shift:Buffer<f32>, scratch:Buffer<f32>, fault:ReadbackBuffer<u32>,
    patches:u32, rows:u32, ready:Option<Revision>,
}

impl JointPatches {
    pub fn disable(&mut self){self.ready=None;self.patches=0;}
    pub fn count(&self)->u32{self.patches}

    /// Construct edges from actual exported joint endpoint scalars, not from
    /// contact-graph multiplicity. Unit pairs already in one baseline domain
    /// need no duplicate factor; all other pairs of a higher-order record are
    /// retained. A world endpoint has no reduced scalar and is already covered
    /// by the baseline domain's diagonal block.
    pub unsafe fn rebuild<D:Device>(&mut self,device:&mut D,model:ModelView,
        joints:FrozenJointBlockView,q:constraint_space::View,owners:&[u32]) -> Result<(),Fault> {
        self.disable();
        if model.revision!=q.revision || model.revision.joints!=joints.revision {
            return Err(refusal("stale matrix, joint export, or constraint projector"));
        }
        let cloth=q.cloth;let bodies=q.bodies;
        let rows=cloth.checked_add(bodies.checked_mul(2).ok_or_else(||refusal("body extent overflow"))?)
            .ok_or_else(||refusal("row extent overflow"))?;
        if model.rows!=rows || owners.len()!=rows as usize {return Err(refusal("invalid row/owner extent"));}
        for body in 0..bodies {
            let row=(cloth+2*body) as usize;
            if owners[row]!=owners[row+1]{return Err(refusal("baseline splits a rigid 6-DOF unit"));}
        }
        let total_ids=(joints.records as usize).checked_mul(24).ok_or_else(||refusal("joint ID extent overflow"))?;
        let counts=constraint_space::read_u32(device,joints.scalar_count,joints.records as usize)?;
        let ids=constraint_space::read_u32(device,joints.scalar_ids,total_ids)?;
        let mut edges=BTreeSet::new();
        for (record,&count) in counts.iter().enumerate() {
            if count>24{return Err(refusal("joint record scalar count exceeds export stride"));}
            let mut units=BTreeSet::new();
            for &scalar in &ids[24*record..24*record+count as usize] {
                let row=scalar/3;
                if row>=rows{return Err(refusal("joint scalar outside reduced system"));}
                units.insert(if row<cloth {row} else {cloth+(row-cloth)/2});
            }
            let u:Vec<_>=units.into_iter().collect();
            for i in 0..u.len(){for j in i+1..u.len(){
                let first=|unit:u32|if unit<cloth {unit} else {cloth+2*(unit-cloth)};
                let a=first(u[i]);let b=first(u[j]);
                if a>=rows || b>=rows{return Err(refusal("joint unit outside reduced system"));}
                if owners[a as usize]!=owners[b as usize] {edges.insert((u[i],u[j]));}
            }}
        }
        if edges.len()>u32::MAX as usize/4{return Err(refusal("patch incidence overflow"));}
        let mut offsets=vec![0u32];let mut members=Vec::new();
        let mut incidence=vec![Vec::<u32>::new();rows as usize];
        let mut degree=vec![0u32;(cloth+bodies) as usize];
        for (patch,(a,b)) in edges.iter().copied().enumerate(){
            for unit in [a,b] {
                degree[unit as usize]=degree[unit as usize].checked_add(1).ok_or_else(||refusal("unit degree overflow"))?;
                let first=if unit<cloth {unit} else {cloth+2*(unit-cloth)};
                let n=if unit<cloth {1} else {2};
                for row in first..first+n {
                    let local=(members.len()-offsets[patch] as usize) as u32;
                    if row>=rows || local>=4{return Err(refusal("invalid patch membership"));}
                    incidence[row as usize].push((patch as u32)*4+local);
                    members.push(row);
                }
            }
            offsets.push(u32::try_from(members.len()).map_err(|_|refusal("membership overflow"))?);
        }
        let mut weights=vec![0.0f32;rows as usize];
        for row in 0..rows {
            let unit=if row<cloth {row} else {cloth+(row-cloth)/2};
            let d=degree[unit as usize];if d!=0{weights[row as usize]=1.0/(d as f32).sqrt();}
        }
        let mut incident_offset=vec![0u32];let mut incident=Vec::new();
        for list in incidence {incident.extend(list);incident_offset.push(u32::try_from(incident.len()).map_err(|_|refusal("incident extent overflow"))?);}
        let n=edges.len();self.rows=rows;
        if n==0 {self.ready=Some(model.revision);return Ok(());}
        let matrix=n.checked_mul(144).ok_or_else(||refusal("factor extent overflow"))?;
        let scratch=n.checked_mul(12).ok_or_else(||refusal("scratch extent overflow"))?;
        if matrix>u32::MAX as usize || scratch>u32::MAX as usize{return Err(refusal("device indexing overflow"));}
        self.member_offset.size(device,offsets.len(),AllocLabel("jp.member_offset"))?;self.member_offset.write(device,0,&offsets)?;
        self.members.size(device,members.len(),AllocLabel("jp.members"))?;self.members.write(device,0,&members)?;
        self.weight.size(device,weights.len(),AllocLabel("jp.weight"))?;self.weight.write(device,0,&weights)?;
        self.incident_offset.size(device,incident_offset.len(),AllocLabel("jp.incident_offset"))?;self.incident_offset.write(device,0,&incident_offset)?;
        self.incident.size(device,incident.len(),AllocLabel("jp.incident"))?;self.incident.write(device,0,&incident)?;
        self.inverse.size(device,matrix,AllocLabel("jp.inverse"))?;
        self.shift.size(device,n,AllocLabel("jp.shift"))?;
        self.scratch.size(device,scratch,AllocLabel("jp.scratch"))?;
        self.fault.size(device,n,AllocLabel("jp.fault"))?;self.fault.seed(device,&vec![0u32;n])?;
        let a=JointPatchFactorArgs {offset:model.offset,column:model.column,value:model.value,
            member_offset:self.member_offset.handle(),members:self.members.handle(),q:q.q,
            inverse:self.inverse.handle(),shift:self.shift.handle(),fault:self.fault.handle(),
            cloth,extent:n as u32,count:n as u32,seam_arena_count:0};
        device.launch("joint.patch.factor",&a,n as u32)?;
        self.fault.download(device)?;
        if let Some((i,c))=self.fault.host().iter().enumerate().find(|(_,c)|**c!=0){
            return Err(refusal(format!("factor patch {i}: fault {c}")));
        }
        self.patches=n as u32;self.ready=Some(model.revision);Ok(())
    }

    pub unsafe fn encode<E:Encoder+?Sized>(&mut self,encoder:&mut E,residual:Handle,result:Handle,revision:Revision)->Result<(),Fault>{
        if self.ready!=Some(revision){return Err(refusal("stale patch factors"));}
        if self.patches==0{return Ok(());}
        let a=JointPatchApplyArgs{member_offset:self.member_offset.handle(),members:self.members.handle(),
            weight:self.weight.handle(),inverse:self.inverse.handle(),residual,scratch:self.scratch.handle(),
            fault:self.fault.handle(),extent:self.patches,count:self.patches,seam_arena_count:0};
        encoder.elements(&a,self.patches)?;
        let a=JointPatchAddArgs{incident_offset:self.incident_offset.handle(),incident:self.incident.handle(),
            scratch:self.scratch.handle(),result,extent:self.rows,count:self.rows,seam_arena_count:0};
        encoder.elements(&a,self.rows)
    }

    pub fn check<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{
        if self.ready.is_none() || self.patches==0{return Ok(());}
        self.fault.download(device)?;
        if let Some((i,c))=self.fault.host().iter().enumerate().find(|(_,c)|**c!=0){
            return Err(refusal(format!("apply patch {i}: fault {c}")));
        } Ok(())
    }
}
