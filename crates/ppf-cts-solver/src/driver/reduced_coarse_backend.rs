//! Resident additive hierarchy; reuses existing local SPD kernels unchanged.
use ppf_cts_compute::{AllocLabel,Buffer,ReadbackBuffer,Device,Encoder,EncoderExt,Fault,Handle};
use super::{kernels::*,reduced_coarse::{self,LevelPlan},
    reduced_model::{partition,Revision,Domains,DENSE_STRIDE},reduced_model_backend::ModelView};
fn refusal(s:impl Into<String>)->Fault {
    Fault::Platform{call:"reduced Galerkin hierarchy",detail:s.into()}
}
#[derive(Clone,Copy)]
pub struct Geometry {
    pub position:Handle,
    pub kind:Handle,
}
#[derive(Default)]
pub struct FrozenGeometry {
    position:Buffer<f32>,kind:Buffer<u32>,fault:ReadbackBuffer<u32>,
    /// Entry resolves every handle before branching; pure FEM still needs an
    /// actual correctly aligned typed allocation, not an absent body handle.
    empty_state:Buffer<crate::data::PdrdRigidState>,
}
impl FrozenGeometry {
    /// vertices is current full-space Vec3f buffer; state is the actual
    /// PdrdRigidState buffer from the fit immediately before frozen assembly.
    pub unsafe fn rebuild<D:Device>(&mut self,d:&mut D,row_to_vertex:Handle,
        vertices:Handle,state:Handle,dof_mask:Handle,cloth:u32,rows:u32)->Result<Geometry,Fault> {
        if cloth>rows || (rows-cloth)%2!=0 || row_to_vertex.size<cloth || state.size<(rows-cloth)/2 {
            return Err(refusal("frozen geometry live handle extents mismatch"));
        }
        // eval_x is a flat f32 buffer (3N elements); mask has N vertex entries.
        if vertices.size%3!=0 || dof_mask.size<vertices.size/3 {
            return Err(refusal("frozen geometry mask extent mismatch"));
        }
        let state=if rows==cloth {
            self.empty_state.size(d,1,AllocLabel("rmc.empty_typed_rigid_state"))?;
            self.empty_state.write(d,0,&[crate::data::PdrdRigidState::default()])?;
            self.empty_state.handle()
        } else {state};
        self.position.size(d,(rows as usize*3).max(1),AllocLabel("rmc.geometry"))?;
        self.kind.size(d,(rows as usize).max(1),AllocLabel("rmc.geometry_kind"))?;
        self.fault.size(d,(rows as usize).max(1),AllocLabel("rmc.geometry_fault"))?;
        let a=VecFillU32Args{array:self.fault.handle(),value:0,count:rows.max(1),seam_arena_count:0};
        d.launch("rmc.geometry_zero",&a,a.count)?;
        if rows!=0 {
            let a=RmcGeometryArgs{row_to_vertex,vertices,state,dof_mask,position:self.position.handle(),
                kind:self.kind.handle(),fault:self.fault.handle(),cloth,extent:rows,count:rows,seam_arena_count:0};
            d.launch("rmc.geometry",&a,rows)?;
        }
        self.fault.download(d)?;
        if self.fault.host().iter().any(|&x|x!=0){return Err(refusal("nonfinite frozen geometry"));}
        Ok(Geometry{position:self.position.handle(),kind:self.kind.handle()})
    }
}
#[derive(Default)]
struct Level {
    owner:Buffer<u32>,member_offset:Buffer<u32>,members:Buffer<u32>,
    offset:Buffer<u32>,column:Buffer<u32>,value:Buffer<f32>,
    position:Buffer<f32>,kind:Buffer<u32>,center:Buffer<f32>,
    modes:Buffer<f32>,p:Buffer<f32>,transform:Buffer<f32>,rank:Buffer<u32>,
    fault:ReadbackBuffer<u32>,rhs:Buffer<f32>,result:Buffer<f32>,
    local:Local,fine_rows:u32,nodes:u32,
    symbolic:Option<super::coarse_assembly_plan::Plan>,
    sym_offsets:Buffer<u32>,sym_reverse:Buffer<u32>,sym_block_pair:Buffer<u32>,sym_records:Buffer<u32>,
}

/// Host-built graph modes used by the opt-in articulation coarse-space
/// prototype. The scalar mode is defined on reduced triplet rows; the three
/// candidate columns use that same graph mode in x/y/z. `active` is per
/// aggregate so an isolated singleton keeps the established geometric modes.
struct GraphModes { row:Vec<f32>, active:Vec<bool> }

/// Find a deterministic lowest non-constant mode of an aggregate's induced
/// reduced-operator graph. This is a structural graph mode, not an eigenvector
/// of the physical Hessian. Current matrix values still define the Galerkin
/// operator and the local factors.
fn build_graph_modes(plan:&LevelPlan,offset:&[u32],column:&[u32],
    multiplicity:Option<&[u32]>) -> Result<GraphModes,Fault> {
    let rows=plan.fine_rows as usize;
    let domains=plan.aggregates.offset.len().saturating_sub(1);
    if plan.aggregates.assignment.len()!=rows || plan.aggregates.offset.len()!=domains+1 ||
       plan.aggregates.offset.first().copied()!=Some(0) ||
       plan.aggregates.offset.windows(2).any(|w|w[0]>w[1]) ||
       plan.aggregates.offset.last().copied().map(|x|x as usize)!=Some(plan.aggregates.members.len()) ||
       offset.len()!=rows+1 || offset.first().copied()!=Some(0) ||
       offset.last().copied().map(|x|x as usize)!=Some(column.len()) ||
       offset.windows(2).any(|w|w[0]>w[1]) ||
       plan.aggregates.assignment.iter().any(|&a|a as usize>=domains) ||
       plan.aggregates.members.iter().any(|&r|r as usize>=rows) ||
       plan.aggregates.members.len()!=rows ||
       column.iter().any(|&c|c as usize>=rows) ||
       multiplicity.is_some_and(|w|w.len()!=column.len()) {
        return Err(refusal("invalid graph-mode source extents"));
    }
    let mut local=vec![usize::MAX;rows];
    let mut graph=GraphModes{row:vec![0.0;rows],active:vec![false;domains]};
    for domain in 0..domains {
        let begin=plan.aggregates.offset[domain] as usize;
        let end=plan.aggregates.offset[domain+1] as usize;
        let members=&plan.aggregates.members[begin..end];
        let n=members.len();
        if n==0 || n>super::reduced_model::CAP {
            return Err(refusal("graph-mode aggregate exceeds reduced-domain contract"));
        }
        for (i,&row) in members.iter().enumerate() {
            let row=row as usize;
            if row>=rows || plan.aggregates.assignment[row] as usize!=domain || local[row]!=usize::MAX {
                return Err(refusal("graph-mode aggregate membership is inconsistent"));
            }
            local[row]=i;
        }
        let mut adjacency=vec![0.0f64;n*n];
        for (i,&row) in members.iter().enumerate() {
            let row=row as usize;
            for k in offset[row] as usize..offset[row+1] as usize {
                let col=column[k] as usize;
                if col==row || plan.aggregates.assignment[col] as usize!=domain { continue; }
                let j=local[col];
                if j==usize::MAX { return Err(refusal("graph-mode local row lookup failed")); }
                let w=f64::from(multiplicity.map_or(1u32,|values|values[k]));
                if !w.is_finite() || w<=0.0 { return Err(refusal("invalid graph-mode edge weight")); }
                // The source CSR is symmetric but may contain additive
                // duplicates. Use one symmetric structural edge weight.
                adjacency[i*n+j]=adjacency[i*n+j].max(w);
                adjacency[j*n+i]=adjacency[j*n+i].max(w);
            }
        }
        let mut laplacian=vec![0.0f64;n*n];
        for i in 0..n {
            let degree=(0..n).map(|j|adjacency[i*n+j]).sum::<f64>();
            laplacian[i*n+i]=degree;
            for j in 0..n { if i!=j {laplacian[i*n+j]=-adjacency[i*n+j];} }
        }
        if let Some(mode)=lowest_nonconstant_symmetric_mode(&laplacian,n) {
            graph.active[domain]=true;
            for (i,&row) in members.iter().enumerate() { graph.row[row as usize]=mode[i]; }
        }
        for &row in members {local[row as usize]=usize::MAX;}
    }
    Ok(graph)
}

/// Symmetric Jacobi eigensolver for the tiny (at most 16x16) aggregate graph.
/// It is used only while building the opt-in coarse space, never in PCG.
fn lowest_nonconstant_symmetric_mode(matrix:&[f64],n:usize)->Option<Vec<f32>> {
    if n<2 || matrix.len()!=n*n || matrix.iter().any(|x|!x.is_finite()) {return None;}
    let mut a=matrix.to_vec();
    let mut vectors=vec![0.0f64;n*n];
    for i in 0..n {vectors[i*n+i]=1.0;}
    let scale=(0..n).map(|i|a[i*n+i].abs()).fold(0.0f64,f64::max).max(1.0);
    let tolerance=1.0e-12*scale;
    for _ in 0..(64*n*n) {
        let mut p=0usize;let mut q=0usize;let mut largest=0.0f64;
        for i in 0..n {for j in i+1..n {
            let candidate=a[i*n+j].abs();
            if candidate>largest {largest=candidate;p=i;q=j;}
        }}
        if largest<=tolerance {break;}
        let apq=a[p*n+q];
        if apq==0.0 {continue;}
        let tau=(a[q*n+q]-a[p*n+p])/(2.0*apq);
        let t=if tau>=0.0 {1.0/(tau+(1.0+tau*tau).sqrt())}
            else {-1.0/(-tau+(1.0+tau*tau).sqrt())};
        let c=1.0/(1.0+t*t).sqrt();let s=t*c;
        let app=a[p*n+p];let aqq=a[q*n+q];
        a[p*n+p]=app-t*apq;a[q*n+q]=aqq+t*apq;
        a[p*n+q]=0.0;a[q*n+p]=0.0;
        for k in 0..n {if k!=p&&k!=q {
            let akp=a[k*n+p];let akq=a[k*n+q];
            let new_p=c*akp-s*akq;let new_q=s*akp+c*akq;
            a[k*n+p]=new_p;a[p*n+k]=new_p;
            a[k*n+q]=new_q;a[q*n+k]=new_q;
        }}
        for k in 0..n {
            let vkp=vectors[k*n+p];let vkq=vectors[k*n+q];
            vectors[k*n+p]=c*vkp-s*vkq;vectors[k*n+q]=s*vkp+c*vkq;
        }
    }
    let mut order=(0..n).collect::<Vec<_>>();
    order.sort_by(|&i,&j|a[i*n+i].partial_cmp(&a[j*n+j]).unwrap_or(std::cmp::Ordering::Equal));
    let eigen_tolerance=1.0e-10*scale;
    let selected=order.into_iter().find(|&j|a[j*n+j]>eigen_tolerance)?;
    let mut mode=(0..n).map(|i|vectors[i*n+selected]).collect::<Vec<_>>();
    let max_abs=mode.iter().map(|x|x.abs()).fold(0.0f64,f64::max);
    if !max_abs.is_finite() || max_abs<=0.0 {return None;}
    for x in &mut mode {*x/=max_abs;}
    if let Some(first)=mode.iter().find(|x|x.abs()>1.0e-12) {if *first<0.0 {for x in &mut mode {*x = -*x;}}}
    mode.iter().all(|x|x.is_finite() && x.abs()<=1.0000001)
        .then(||mode.into_iter().map(|x|x as f32).collect())
}

fn apply_graph_modes_to_candidates(values:&mut[f32],kinds:&[u32],owner:&[u32],graph:&GraphModes)
    ->Result<(),Fault> {
    if values.len()!=kinds.len().checked_mul(18).ok_or_else(||refusal("graph-mode candidate extent overflow"))? ||
       owner.len()!=kinds.len() || graph.row.len()!=kinds.len() {
        return Err(refusal("graph-mode candidate buffer extent mismatch"));
    }
    for row in 0..kinds.len() {
        let domain=owner[row] as usize;
        if domain>=graph.active.len() {return Err(refusal("graph-mode owner is outside aggregate table"));}
        if !graph.active[domain] || kinds[row]==3 {continue;}
        for component in 0..3 {
            for candidate in 0..3 {
                values[18*row+6*component+3+candidate]=
                    if component==candidate {graph.row[row]} else {0.0};
            }
        }
    }
    Ok(())
}

/// Replace the three geometric rotation candidates by graph-varying triplet
/// modes for connected aggregates. It deliberately leaves the three global
/// translation modes intact. Pinned rows remain zero. The feature is opt-in
/// because the six-column kernel ABI cannot add rank without changing the
/// generated entry contract; callers can revert by unsetting the environment
/// switch.
unsafe fn install_graph_enrichment<D:Device>(device:&mut D,plan:&LevelPlan,
    offset:&[u32],column:&[u32],multiplicity:Option<&[u32]>,kind:Handle,modes:Handle)
    ->Result<(),Fault> {
    let graph=build_graph_modes(plan,offset,column,multiplicity)?;
    if !graph.active.iter().any(|&active|active) {return Ok(());}
    let rows=plan.fine_rows as usize;
    let mode_values=rows.checked_mul(18).ok_or_else(||refusal("graph-mode extent overflow"))?;
    if kind.size<rows as u32 || mode_values>u32::MAX as usize || modes.size<mode_values as u32 {
        return Err(refusal("graph-mode device buffer extent mismatch"));
    }
    let mut kinds=vec![0u32;rows];
    let mut kind_bytes=unsafe {std::slice::from_raw_parts_mut(kinds.as_mut_ptr().cast::<u8>(),rows*std::mem::size_of::<u32>())};
    device.read(kind,0,kind_bytes)?;
    let mut values=vec![0.0f32;mode_values];
    let mut value_bytes=unsafe {std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(),values.len()*std::mem::size_of::<f32>())};
    device.read(modes,0,value_bytes)?;
    apply_graph_modes_to_candidates(&mut values,&kinds,&plan.aggregates.assignment,&graph)?;
    let bytes=unsafe {std::slice::from_raw_parts(values.as_ptr().cast::<u8>(),values.len()*std::mem::size_of::<f32>())};
    device.write(modes,0,bytes)
}

#[derive(Default)]
pub struct Hierarchy {levels:Vec<Level>,ready:Option<Revision>}
impl Hierarchy {
    /// Both graph slices describe the exact same live fine CSR as fine.
    /// Geometry is live frozenNewton world geometry in fine reduced row order.
    /// A rebuild error invalidates the hierarchy. Caller may select fine-only
    /// BEFORE PCG for rank64; record that decision and keep it fixed.
    pub unsafe fn rebuild<D:Device>(&mut self,device:&mut D,fine:ModelView,
        cloth:u32,bodies:u32,offset:&[u32],column:&[u32],weights:&[u32],geometry:Geometry,
        max_levels:u32,constraints:super::constraint_space::View)->Result<(),Fault> {
        self.ready=None;
        if constraints.revision!=fine.revision{return Err(refusal("stale hierarchy constraints"));}
        if u64::from(fine.rows)!=u64::from(cloth)+2*u64::from(bodies) {return Err(refusal("fine row layout mismatch"));}
        let plans=reduced_coarse::hierarchy_weighted(cloth,bodies,offset,column,max_levels,Some(weights)).map_err(refusal)?;
        let graph_enrichment=std::env::var_os("PPF_COARSE_GRAPH_ENRICH").is_some_and(|x|x=="1");
        // Retiring levels must release the added symbolic device allocations.
        for level in self.levels.iter_mut().skip(plans.len()) {level.release_symbolic(device)?;}
        self.levels.resize_with(plans.len(),Level::default);
        let symbolic_budget=if std::env::var("PPF_COARSE_SYMBOLIC_PLAN").ok().as_deref()==Some("1") {
            std::env::var("PPF_COARSE_SYMBOLIC_BUDGET_BYTES").ok().and_then(|v|v.parse::<usize>().ok()).filter(|&v|v>0)
        } else {None};
        let mut input=fine;
        for (index,plan) in plans.iter().enumerate() {
            let (fine_off,fine_col)=if index==0 {(offset,column)} else {(&plans[index-1].offset[..],&plans[index-1].column[..])};
            let resident=self.levels.iter().fold(0usize,|n,l|n.saturating_add(l.symbolic_bytes()));
            let (before,tail)=self.levels.split_at_mut(index);
            let level=&mut tail[0];
            let use_symbolic=if let Some(budget)=symbolic_budget {
                level.prepare_symbolic(device,fine_off,fine_col,plan,budget,resident)?
            } else {false};
            level.allocate(device,plan)?;
            let g=if index==0 {geometry} else {
                let child=&before[index-1];
                let a=RmcChildGeometryArgs{center:child.center.handle(),position:level.position.handle(),
                    kind:level.kind.handle(),extent:level.fine_rows,count:level.fine_rows,seam_arena_count:0};
                device.launch("rmc.child_geometry",&a,level.fine_rows)?;
                Geometry{position:level.position.handle(),kind:level.kind.handle()}
            };
            let a=RmcCentersArgs{member_offset:level.member_offset.handle(),members:level.members.handle(),
                position:g.position,kind:g.kind,center:level.center.handle(),fault:level.fault.handle(),
                extent:level.nodes,count:level.nodes,seam_arena_count:0};
            device.launch("rmc.centers",&a,level.nodes)?;
            if index==0 {
                let a=RmcModesArgs{owner:level.owner.handle(),kind:g.kind,position:g.position,
                    center:level.center.handle(),modes:level.modes.handle(),extent:level.fine_rows,
                    count:level.fine_rows,seam_arena_count:0};
                device.launch("rmc.modes",&a,level.fine_rows)?;
                if graph_enrichment {
                    install_graph_enrichment(device,plan,fine_off,fine_col,
                        (index==0).then_some(weights),g.kind,level.modes.handle())?;
                }
                let units=cloth.checked_add(bodies).ok_or_else(||refusal("constraint units overflow"))?;
                let a=RqcModesArgs{owner:level.owner.handle(),q:constraints.q,modes:level.modes.handle(),
                    fault:level.fault.handle(),cloth,bodies,extent:units,count:units,seam_arena_count:0};
                device.launch("rqc.modes",&a,units)?;
                level.check(device)?;
            } else {
                let child=&before[index-1];
                let a=RmcParentModesArgs{owner:level.owner.handle(),child_center:child.center.handle(),
                    parent_center:level.center.handle(),child_r:child.transform.handle(),
                    modes:level.modes.handle(),extent:level.fine_rows,count:level.fine_rows,seam_arena_count:0};
                device.launch("rmc.parent_modes",&a,level.fine_rows)?;
                if graph_enrichment {
                    install_graph_enrichment(device,plan,fine_off,fine_col,None,g.kind,level.modes.handle())?;
                }
            }
            let a=RmcBasisArgs{member_offset:level.member_offset.handle(),members:level.members.handle(),
                offset:input.offset,column:input.column,value:input.value,modes:level.modes.handle(),
                p:level.p.handle(),transform:level.transform.handle(),rank:level.rank.handle(),
                fault:level.fault.handle(),extent:level.nodes,count:level.nodes,seam_arena_count:0};
            device.launch("rmc.basis",&a,level.nodes)?;
            level.check(device)?;
            if use_symbolic {
                let a=RmcContributionGalerkinArgs{pair_offsets:level.sym_offsets.handle(),reverse:level.sym_reverse.handle(),
                    block_pair:level.sym_block_pair.handle(),records:level.sym_records.handle(),fine_value:input.value,
                    p:level.p.handle(),rank:level.rank.handle(),offset:level.offset.handle(),column:level.column.handle(),
                    value:level.value.handle(),fault:level.fault.handle(),extent:plan.rows(),count:plan.rows(),seam_arena_count:0};
                device.launch("rmc.contribution_galerkin",&a,plan.rows())?;
            } else {
            let a=RmcGalerkinArgs{member_offset:level.member_offset.handle(),members:level.members.handle(),
                owner:level.owner.handle(),fine_offset:input.offset,fine_column:input.column,
                fine_value:input.value,p:level.p.handle(),rank:level.rank.handle(),offset:level.offset.handle(),
                column:level.column.handle(),value:level.value.handle(),fault:level.fault.handle(),
                extent:plan.rows(),count:plan.rows(),seam_arena_count:0};
            device.launch("rmc.galerkin",&a,plan.rows())?;
            }
            level.check(device)?;
            if index==0 && super::constraint_space::diagnostic_selected(fine.revision)?{
                level.constraint_diagnostic(device,input,constraints)?;
            }
            let view=ModelView{rows:plan.rows(),blocks:plan.column.len() as u32,
                offset:level.offset.handle(),column:level.column.handle(),value:level.value.handle(),revision:fine.revision};
            let domains=partition(0,plan.rows()/2,&plan.offset,&plan.column).map_err(refusal)?;
            level.local.factor(device,view,&domains)?;
            input=view;
        }
        self.ready=Some(fine.revision);Ok(())
    }
    /// Caller first encodes fine localSPD result=S r. This adds the complete
    /// transported hierarchy, then caller applies original Q to total result.
    pub unsafe fn encode_add<E:Encoder+?Sized>(&mut self,encoder:&mut E,
        residual:Handle,result:Handle,revision:Revision)->Result<(),Fault> {
        if self.ready!=Some(revision){return Err(refusal("stale hierarchy"));}
        let mut source=residual;
        for level in &mut self.levels {
            let a=RmcRestrictArgs{member_offset:level.member_offset.handle(),members:level.members.handle(),
                p:level.p.handle(),fine:source,rank:level.rank.handle(),coarse:level.rhs.handle(),
                fault:level.fault.handle(),extent:level.nodes,count:level.nodes,seam_arena_count:0};
            encoder.elements(&a,level.nodes)?;
            level.local.encode(encoder,level.rhs.handle(),level.result.handle())?;
            source=level.rhs.handle();
        }
        for index in (0..self.levels.len()).rev() {
            let target=if index==0 {result} else {self.levels[index-1].result.handle()};
            let level=&mut self.levels[index];
            let a=RmcProlongAddArgs{owner:level.owner.handle(),p:level.p.handle(),coarse:level.result.handle(),
                fine:target,fault:level.fault.handle(),extent:level.fine_rows,
                count:level.fine_rows,seam_arena_count:0};
            encoder.elements(&a,level.fine_rows)?;
        }
        Ok(())
    }
    pub fn check<D:Device>(&mut self,device:&mut D)->Result<(),Fault> {
        if self.ready.is_none(){return Err(refusal("unready hierarchy"));}
        for level in &mut self.levels {level.check(device)?;level.local.check(device)?;}
        Ok(())
    }
    pub fn levels(&self)->usize {self.levels.len()}
}
#[derive(Default)]
struct Local {
    member_offset:Buffer<u32>,members:Buffer<u32>,dense:Buffer<f32>,
    normalized:Buffer<f32>,lower:Buffer<f32>,inverse:Buffer<f32>,scale:Buffer<f32>,
    shift:ReadbackBuffer<f32>,fault:ReadbackBuffer<u32>,receipt:ReadbackBuffer<u32>,domains:u32,
}
impl Local {
    unsafe fn factor<D:Device>(&mut self,d:&mut D,view:ModelView,domains:&Domains)->Result<(),Fault> {
        self.domains=(domains.offset.len()-1) as u32;
        let n=self.domains as usize;
        if n>u32::MAX as usize/DENSE_STRIDE{return Err(refusal("coarse local extent overflow"));}
        self.member_offset.size(d,domains.offset.len(),AllocLabel("rmc.local_offsets"))?;
        self.member_offset.write(d,0,&domains.offset)?;
        self.members.size(d,domains.members.len().max(1),AllocLabel("rmc.local_members"))?;
        self.members.write(d,0,&domains.members)?;
        for b in [&mut self.dense,&mut self.normalized,&mut self.lower,&mut self.inverse] {
            b.size(d,(n*DENSE_STRIDE).max(1),AllocLabel("rmc.local_dense"))?;
        }
        self.scale.size(d,(48*n).max(1),AllocLabel("rmc.local_scale"))?;
        self.shift.size(d,n.max(1),AllocLabel("rmc.local_shift"))?;
        self.fault.size(d,n.max(1),AllocLabel("rmc.local_fault"))?;
        self.receipt.size(d,4,AllocLabel("rmc.local_receipt"))?;
        for (array,count) in [(self.fault.handle(),self.domains.max(1)),(self.receipt.handle(),4)] {
            let a=VecFillU32Args{array,value:0,count,seam_arena_count:0};d.launch("rmc.local_zero",&a,count)?;
        }
        let a=RsmGatherArgs{offset:view.offset,column:view.column,value:view.value,
            member_offset:self.member_offset.handle(),members:self.members.handle(),dense:self.dense.handle(),
            fault:self.fault.handle(),extent:self.domains,count:self.domains,seam_arena_count:0};
        d.launch("rmc.local_gather",&a,self.domains)?;
        let a=RsmFactorArgs{member_offset:self.member_offset.handle(),dense:self.dense.handle(),
            normalized:self.normalized.handle(),lower:self.lower.handle(),inverse:self.inverse.handle(),
            scale:self.scale.handle(),shift:self.shift.handle(),fault:self.fault.handle(),
            extent:self.domains,count:self.domains,seam_arena_count:0};
        d.launch("rmc.local_factor",&a,self.domains)?;
        self.fault.download(d)?;
        if self.fault.host().iter().any(|&x|x!=0){return Err(refusal("coarse local factor refusal"));}
        self.shift.download(d)?;
        if self.shift.host().iter().any(|&x|!x.is_finite()||x<0.0||x>1.001e-4) {
            return Err(refusal("coarse local factor jitter invalid"));
        }
        Ok(())
    }
    unsafe fn encode<E:Encoder+?Sized>(&mut self,e:&mut E,residual:Handle,result:Handle)->Result<(),Fault> {
        let a=RsmApplyArgs{member_offset:self.member_offset.handle(),members:self.members.handle(),
            inverse:self.inverse.handle(),residual,result,apply_fault:self.fault.handle(),receipt:self.receipt.handle(),
            extent:self.domains,count:self.domains,seam_arena_count:0};
        e.elements(&a,self.domains)
    }
    fn check<D:Device>(&mut self,d:&mut D)->Result<(),Fault> {
        self.receipt.download(d)?;let r=self.receipt.host();
        if r[2]!=0||r[3]!=0||u64::from(r[1])!=u64::from(r[0])*u64::from(self.domains) {
            return Err(refusal("coarse local apply receipt mismatch/fault"));
        }
        Ok(())
    }
}
impl Level {
    fn release_symbolic<D:Device>(&mut self,d:&mut D)->Result<(),Fault> {
        self.symbolic=None;
        for b in [&mut self.sym_offsets,&mut self.sym_reverse,&mut self.sym_block_pair,&mut self.sym_records] {
            if !b.handle().is_none(){b.free(d)?;}
        }
        Ok(())
    }
    fn symbolic_bytes(&self)->usize {
        let host=self.symbolic.as_ref().map_or(0,|p|p.host_bytes());
        [&self.sym_offsets,&self.sym_reverse,&self.sym_block_pair,&self.sym_records]
            .iter().fold(host,|n,b|n.saturating_add((b.handle().allocated as usize).saturating_mul(4)))
    }
    fn prepare_symbolic<D:Device>(&mut self,d:&mut D,fo:&[u32],fc:&[u32],plan:&LevelPlan,
        budget:usize,resident:usize)->Result<bool,Fault> {
        use super::coarse_assembly_plan::{self,Outcome};
        if resident>budget || fc.len()>(u32::MAX/4) as usize {return Ok(false);}
        if self.symbolic.as_ref().map_or(false,|p|p.matches(fo,fc,plan)) {return Ok(true);}
        let Some(peak)=coarse_assembly_plan::peak_upper(fo,fc,plan) else{return Ok(false)};
        if resident.checked_add(peak).map_or(true,|n|n>budget){return Ok(false);}
        let next=match coarse_assembly_plan::build(fo,fc,plan) {
            Outcome::Ready(p)=>p,Outcome::Resource=>return Ok(false),
            Outcome::Invalid(detail)=>return Err(refusal(detail)),
        };
        if next.host_bytes()>peak {return Ok(false);}
        // Device faults propagate. Clear old plan before writes so a partial
        // upload cannot be selected in a later Newton step.
        self.symbolic=None;
        for (buffer,data) in [(&mut self.sym_offsets,&next.pair_offsets),(&mut self.sym_reverse,&next.reverse),
            (&mut self.sym_block_pair,&next.block_pair),(&mut self.sym_records,&next.records)] {
            buffer.size(d,data.len().max(1),AllocLabel("rmc.symbolic"))?;
            if !data.is_empty(){buffer.write(d,0,data)?;}
        }
        self.symbolic=Some(next);Ok(true)
    }
    fn allocate<D:Device>(&mut self,d:&mut D,plan:&LevelPlan)->Result<(),Fault> {
        self.fine_rows=plan.fine_rows;self.nodes=plan.rows()/2;
        let rows=plan.fine_rows as usize;let nodes=self.nodes as usize;
        if rows>u32::MAX as usize/18 || nodes>u32::MAX as usize/36 {
            return Err(refusal("coarse numeric indexing exceeds u32"));
        }
        for (buffer,data) in [(&mut self.owner,&plan.aggregates.assignment),
            (&mut self.member_offset,&plan.aggregates.offset),(&mut self.members,&plan.aggregates.members),
            (&mut self.offset,&plan.offset),(&mut self.column,&plan.column)] {
            buffer.size(d,data.len().max(1),AllocLabel("rmc.graph"))?;buffer.write(d,0,data)?;
        }
        for (buffer,count) in [(&mut self.value,9*plan.column.len()),(&mut self.position,3*rows),
            (&mut self.center,3*nodes),(&mut self.modes,18*rows),(&mut self.p,18*rows),
            (&mut self.transform,36*nodes),(&mut self.rhs,6*nodes),(&mut self.result,6*nodes)] {
            buffer.size(d,count.max(1),AllocLabel("rmc.numeric"))?;
        }
        self.kind.size(d,rows.max(1),AllocLabel("rmc.kind"))?;
        self.rank.size(d,nodes.max(1),AllocLabel("rmc.rank"))?;
        self.fault.size(d,rows.max(nodes).max(1),AllocLabel("rmc.fault"))?;
        let a=VecFillU32Args{array:self.fault.handle(),value:0,count:rows.max(nodes).max(1) as u32,seam_arena_count:0};
        unsafe{d.launch("rmc.zero_fault",&a,a.count)?;} Ok(())
    }
    fn check<D:Device>(&mut self,d:&mut D)->Result<(),Fault> {
        self.fault.download(d)?;
        if let Some((i,c))=self.fault.host().iter().enumerate().find(|(_,c)|**c!=0) {
            if *c==641 || *c==642 {
                let cause=if *c==641 {"weighted mode underflow"} else {"squared residual underflow"};
                return Err(Fault::Platform{call:"reduced coarse rank ambiguity",detail:format!("coarse fault{c} at {i}: {cause}")});
            }
            return Err(refusal(format!("coarse fault {c} at {i}")));
        }
        Ok(())
    }
}

impl Level{
 unsafe fn constraint_diagnostic<D:Device>(&mut self,d:&mut D,fine:ModelView,qv:super::constraint_space::View)->Result<(),Fault>{
 use super::constraint_space::{read_f32,read_u32,q_entry};
 if fine.rows>512||fine.blocks>100000{return Err(refusal("opt-in coarse constraint diagnostic bounded rows512/blocks100000"));}
 let rows=fine.rows as usize;let dim=3*rows;let owner=read_u32(d,self.owner.handle(),rows)?;
 let q=read_f32(d,qv.q,36*(qv.cloth+qv.bodies)as usize)?;let p=read_f32(d,self.p.handle(),18*rows)?;
 let mut qp_error=0f64;let mut pscale=0f64;
 for i in 0..dim{for c in 0..6{let mut a=0f64;for j in 0..dim{if owner[j/3]==owner[i/3]{a+=q_entry(&q,qv.cloth,(i/3)as u32,i%3,(j/3)as u32,j%3)*f64::from(p[6*j+c]);}}qp_error=qp_error.max((a-f64::from(p[6*i+c])).abs());pscale=pscale.max(f64::from(p[6*i+c]).abs());}}
 let fully_constrained_rows_exact_zero=(0..rows).all(|row|{
   let unit=if row<qv.cloth as usize{row}else{qv.cloth as usize+(row-qv.cloth as usize)/2};
   let axis=if row<qv.cloth as usize{0}else{3*((row-qv.cloth as usize)%2)};
   let zero=(axis..axis+3).all(|i|(0..6).all(|j|q[36*unit+6*i+j]==0.));
   !zero || p[18*row..18*(row+1)].iter().all(|&x|x==0.)});
 if !fully_constrained_rows_exact_zero{return Err(refusal("fully constrained P row must be exactly zero"));}
 let bound=4096.*f64::from(f32::EPSILON)*pscale;if !qp_error.is_finite()||qp_error>bound{return Err(refusal(format!("Q P span invariant error {qp_error} bound {bound}")));}
 let rank=read_u32(d,self.rank.handle(),self.nodes as usize)?;
 let y:Vec<f64>=(0..6*self.nodes as usize).map(|i|if i%6<rank[i/6]as usize{(((5*i)%13)as f64-6.)/16.}else{0.}).collect();
 let z:Vec<f64>=(0..dim).map(|i|(0..6).map(|j|f64::from(p[6*i+j])*y[6*owner[i/3]as usize+j]).sum()).collect();
 let fo=read_u32(d,fine.offset,rows+1)?;let fc=read_u32(d,fine.column,fine.blocks as usize)?;let fv=read_f32(d,fine.value,9*fine.blocks as usize)?;
 let co=read_u32(d,self.offset.handle(),2*self.nodes as usize+1)?;let nb=*co.last().unwrap()as usize;let cc=read_u32(d,self.column.handle(),nb)?;let cv=read_f32(d,self.value.handle(),9*nb)?;
 let mut ef=0f64;let mut ec=0f64;let mut absolute=0f64;
 for r in 0..rows{for k in fo[r]as usize..fo[r+1]as usize{for i in 0..3{for j in 0..3{let t=z[3*r+i]*f64::from(fv[9*k+3*j+i])*z[3*fc[k]as usize+j];ef+=t;absolute+=t.abs();}}}}
 for r in 0..2*self.nodes as usize{for k in co[r]as usize..co[r+1]as usize{for i in 0..3{for j in 0..3{let t=y[3*r+i]*f64::from(cv[9*k+3*j+i])*y[3*cc[k]as usize+j];ec+=t;absolute+=t.abs();}}}}
 let energy_bound=(8.*fine.blocks as f64+128.)*f64::from(f32::EPSILON)*absolute;let error=(ef-ec).abs();if !error.is_finite()||error>energy_bound{return Err(refusal(format!("constrained Galerkin energy error {error} bound {energy_bound}")));}
 eprintln!("SCHWARZ_CONSTRAINT_COARSE_CHECK {}",serde_json::json!({"revision":qv.revision.operator,"fully_constrained_rows_exact_zero":fully_constrained_rows_exact_zero,"QP_minus_P_max":qp_error,"QP_bound":bound,"rank_zero_nodes":rank.iter().filter(|&&r|r==0).count(),"fine_energy":ef,"Galerkin_energy":ec,"energy_error":error,"energy_roundoff_bound":energy_bound,"scope":"same stored frozen unique B and first-level P; deterministic mixed probe; inactive rank padding excluded"}));Ok(())
 }
}

#[cfg(test)]
mod graph_mode_tests {
    use super::{apply_graph_modes_to_candidates,build_graph_modes,lowest_nonconstant_symmetric_mode,GraphModes};
    use super::super::{reduced_coarse::LevelPlan,reduced_model::Domains};

    fn plan(rows:u32,domains:Domains)->LevelPlan {
        LevelPlan{fine_rows:rows,aggregates:domains,offset:Vec::new(),column:Vec::new()}
    }

    #[test]
    fn graph_mode_is_finite_normalized_nonconstant_and_deterministic() {
        let p=plan(4,Domains{assignment:vec![0;4],offset:vec![0,4],members:vec![0,1,2,3]});
        let offset=[0,1,3,5,6];
        let column=[1,0,2,1,3,2];
        let multiplicity=[1;6];
        let first=build_graph_modes(&p,&offset,&column,Some(&multiplicity)).expect("valid path graph");
        let second=build_graph_modes(&p,&offset,&column,Some(&multiplicity)).expect("repeatable path graph");
        assert!(first.active[0]);
        assert_eq!(first.row,second.row,"deterministic sign and mode ordering");
        assert!(first.row.iter().all(|x|x.is_finite()&&x.abs()<=1.000001));
        assert!((first.row.iter().map(|&x|f64::from(x)).sum::<f64>()).abs()<1.0e-5);
        assert!(first.row.windows(2).any(|w|(w[0]-w[1]).abs()>0.1),"mode must vary across joint-graph rows");
    }

    #[test]
    fn disconnected_graph_retains_a_finite_nonconstant_component_mode() {
        let p=plan(4,Domains{assignment:vec![0;4],offset:vec![0,4],members:vec![0,1,2,3]});
        let offset=[0,1,2,3,4];
        let column=[1,0,3,2];
        let modes=build_graph_modes(&p,&offset,&column,None).expect("valid disconnected graph");
        assert!(modes.active[0]);
        assert!(modes.row.iter().all(|x|x.is_finite()&&x.abs()<=1.000001));
        assert!(modes.row.windows(2).any(|w|(w[0]-w[1]).abs()>0.1));
    }

    #[test]
    fn isolated_aggregate_keeps_the_legacy_geometric_basis() {
        let p=plan(1,Domains{assignment:vec![0],offset:vec![0,1],members:vec![0]});
        let modes=build_graph_modes(&p,&[0,0],&[],None).expect("valid isolated row");
        assert!(!modes.active[0]);
        assert_eq!(modes.row,[0.0]);
        assert!(lowest_nonconstant_symmetric_mode(&[0.0],1).is_none());
    }

    #[test]
    fn enrichment_preserves_translations_pinned_rows_and_unselected_domains() {
        let graph=GraphModes{row:vec![-1.0,1.0,0.0],active:vec![true,false]};
        let owner=[0,0,1];
        let kinds=[0,3,1];
        let mut values=vec![2.0f32;54];
        values[18..36].fill(0.0);
        apply_graph_modes_to_candidates(&mut values,&kinds,&owner,&graph).expect("valid candidate extent");
        for row in [0usize,2] {
            for component in 0..3 {
                for candidate in 0..3 {
                    assert_eq!(values[18*row+6*component+candidate],2.0,"translation candidates stay intact");
                }
            }
        }
        for component in 0..3 {
            for candidate in 0..3 {
                assert_eq!(values[6*component+3+candidate],if component==candidate{-1.0}else{0.0});
                assert_eq!(values[18+6*component+3+candidate],0.0,"pinned row stays zero");
                assert_eq!(values[36+6*component+3+candidate],2.0,"inactive aggregate retains its legacy modes");
            }
        }
    }
}



