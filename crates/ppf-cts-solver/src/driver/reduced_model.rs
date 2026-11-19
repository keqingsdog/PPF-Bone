//! Glue-neutral orchestration and constrained graph aggregation.
//! The adapter owns device buffers/args; no existing kernel IDs are assigned here.
use std::collections::BTreeMap;

pub const CAP: usize = 16;
pub const DENSE_STRIDE: usize = 48 * 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Revision {
    pub operator: u64,
    pub mapping: u64,
    pub joints: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub cloth: u32,
    pub bodies: u32,
    pub records: u32,
    /// Sum of ALL full operator CSR block spans plus full source rows for C (unfiltered upper bound).
    pub source_blocks: u64,
}

impl Layout {
    pub fn checked_extents(self) -> Result<(u32, usize), String> {
        let rows = u64::from(self.cloth) + 2 * u64::from(self.bodies);
        if rows > u64::from(u32::MAX / 3) {
            return Err("reduced scalar layout overflows u32".into());
        }
        // General schema permits 24 distinct triplets; a valid normal record
        // usually has <=8. Do not size on that undocumented favorable shape.
        // Each full operator block projects to at most four triplet blocks.
        // source_blocks also budgets full body C-diagonal terms.
        let capacity = self.source_blocks.checked_mul(4).and_then(|x|x.checked_add(u64::from(self.cloth)))
            .and_then(|x| x.checked_add(4 * u64::from(self.bodies)))
            .and_then(|x| x.checked_add(576 * u64::from(self.records)))
            .ok_or_else(|| "reduced CSR capacity overflows u64".to_string())?;
        // All neutral float indexing is u32 including 9*block.
        if capacity > u64::from(u32::MAX / 9) {
            return Err("reduced CSR value indexing exceeds u32".into());
        }
        Ok((rows as u32, capacity as usize))
    }
}

#[derive(Debug, Clone)]
pub struct Domains {
    pub assignment: Vec<u32>,
    pub offset: Vec<u32>,
    pub members: Vec<u32>,
}

/// HEM over B's CSR *structure*. Numeric coefficients remain on device.
/// Duplicates represent additive model contributions and count as structural
/// edge weight; self entries do not create graph edges. Body row pairs start as
/// indivisible units. Repeated pairwise heavy-edge matching grows units up to16.
pub fn partition(
    cloth: u32, bodies: u32, offset: &[u32], column: &[u32],
) -> Result<Domains, String> {
    partition_weighted(cloth,bodies,offset,column,None)
}

/// Multiplicity is EXACT original duplicate count, preserving original HEM
/// edge weights while numerical CSR stores one sum per (row,column).
pub fn partition_weighted(cloth:u32,bodies:u32,offset:&[u32],column:&[u32],weight:Option<&[u32]>)
    ->Result<Domains,String>{
    if let Some(w)=weight{if w.len()!=column.len()||w.iter().any(|&x|x==0){return Err("invalid exact CSR multiplicities".into());}}

    let rows64 = u64::from(cloth) + 2 * u64::from(bodies);
    let rows = usize::try_from(rows64).map_err(|_| "rows overflow")?;
    if rows > u32::MAX as usize || offset.len() != rows + 1 || offset[0] != 0
        || offset.windows(2).any(|p| p[0] > p[1])
        || offset[rows] as usize != column.len()
        || column.iter().any(|&c| c as usize >= rows) {
        return Err("invalid reduced model graph CSR".into());
    }
    let mut groups: Vec<Vec<u32>> = (0..cloth).map(|r| vec![r]).collect();
    groups.extend((0..bodies).map(|b| vec![cloth+2*b, cloth+2*b+1]));
    loop {
        let mut owner=vec![0usize;rows];
        for (g,members) in groups.iter().enumerate() {
            for &r in members {owner[r as usize]=g;}
        }
        let mut edges: Vec<BTreeMap<usize,u64>>=vec![BTreeMap::new();groups.len()];
        for r in 0..rows {
            for k in offset[r] as usize..offset[r+1] as usize {
                let c=column[k];let w=weight.map_or(1u64,|v|u64::from(v[k]));
                let (a,b)=(owner[r],owner[c as usize]);
                if a!=b {*edges[a].entry(b).or_default()+=w;*edges[b].entry(a).or_default()+=w;}
            }
        }
        let mut consumed=vec![false;groups.len()];
        let mut next=Vec::new();
        let mut merged=false;
        for a in 0..groups.len() {
            if consumed[a] {continue;}
            consumed[a]=true;
            let mut best=None;
            for (&b,&weight) in &edges[a] {
                if consumed[b] || groups[a].len()+groups[b].len()>CAP {continue;}
                if best.map_or(true, |(old,old_weight)| weight>old_weight || (weight==old_weight && b<old)) {
                    best=Some((b,weight));
                }
            }
            let mut members=groups[a].clone();
            if let Some((b,_))=best {consumed[b]=true;members.extend(&groups[b]);merged=true;}
            members.sort_unstable();
            next.push(members);
        }
        groups=next;
        if !merged {break;}
    }
    let mut result=Domains{assignment:vec![u32::MAX;rows],offset:vec![0],members:Vec::with_capacity(rows)};
    for (g,members) in groups.iter().enumerate() {
        for &r in members {result.assignment[r as usize]=g as u32;result.members.push(r);}
        result.offset.push(result.members.len() as u32);
    }
    Ok(result)
}

/// Backend contract intentionally typed by operations rather than invented IDs.
/// Every operation completes before the next dependent operation, or encodes
/// them with equivalent queue ordering. All numerical buffers live on device.
pub trait Backend {
    type Error;
    fn allocate(&mut self, rows: u32, capacity: usize, records: u32) -> Result<(),Self::Error>;
    /// Materialize partition-independent SPD body seeds BEFORE any CSR fill.
    /// Raw body input excludes material joints. Base fill reads these seeds.
    fn body_seed(&mut self, bodies: u32) -> Result<(),Self::Error>;
    /// Device base_count; initialize row_count[rows]=0 too.
    fn base_count(&mut self, layout: Layout) -> Result<(),Self::Error>;
    fn joint_count(&mut self, records: u32, rows: u32) -> Result<(),Self::Error>;
    /// Download/reduce fault flags ONLY; failure must prevent subsequent fill.
    fn check_fault(&mut self, extent: u32) -> Result<(),Self::Error>;
    /// Existing ScanScratch.exclusive(row_count, rows), including final total.
    fn scan_counts(&mut self, rows: u32) -> Result<(),Self::Error>;
    fn base_fill(&mut self, layout: Layout) -> Result<(),Self::Error>;
    fn joint_fill(&mut self, records: u32) -> Result<(),Self::Error>;
    fn validate_csr(&mut self, rows: u32) -> Result<(),Self::Error>;
    /// Allowed graph lifecycle: offsets and column IDs, NEVER numerical values.
    fn graph(&mut self, rows: u32) -> Result<(Vec<u32>,Vec<u32>),Self::Error>;
    fn graph_weights(&self)->Option<&[u32]>{None}
    fn upload_domains(&mut self, domains: &Domains) -> Result<(),Self::Error>;
    fn gather(&mut self, domains: u32) -> Result<(),Self::Error>;
    fn factor(&mut self, domains: u32) -> Result<(),Self::Error>;
    /// Reduced projector on source into a separate scratch residual buffer.
    fn project_input(&mut self) -> Result<(),Self::Error>;
    fn apply(&mut self, domains: u32) -> Result<(),Self::Error>;
    /// Original reduced projector on the computed output buffer.
    fn project_output(&mut self) -> Result<(),Self::Error>;
}

#[derive(Debug)]
pub enum Error<E> { Invalid(String), Device(E) }

#[derive(Default)]
pub struct State {
    ready: Option<Revision>,
    domains: u32,
    graph_partition: Option<GraphPartition>,
    partition_reused: bool,
    partition_ms: f64,
}

/// This is topology only: never cache numerical coefficients, factors, or Q.
/// Exact graph identity is needed because contacts can change without changing
/// array lengths or the number of rows.
struct GraphPartition {
    cloth: u32,
    bodies: u32,
    offset: Vec<u32>,
    column: Vec<u32>,
    weight: Option<Vec<u32>>,
    domains: Domains,
}

impl State {
    pub fn is_current(&self, revision: Revision) -> bool { self.ready == Some(revision) }
    pub fn domains(&self) -> u32 { self.domains }
    pub fn partition_stats(&self)->(bool,f64){(self.partition_reused,self.partition_ms)}
    pub fn rebuild<B: Backend>(&mut self, backend: &mut B, layout: Layout, revision: Revision)
        -> Result<(),Error<B::Error>> {
        self.ready=None; // any failed rebuild invalidates the previous factors
        let (rows,capacity)=layout.checked_extents().map_err(Error::Invalid)?;
        backend.allocate(rows,capacity,layout.records).map_err(Error::Device)?;
        if rows==0 {self.domains=0;self.ready=Some(revision);return Ok(());}
        if layout.bodies!=0 {
            backend.body_seed(layout.bodies).map_err(Error::Device)?;
            backend.check_fault(layout.bodies).map_err(Error::Device)?;
        }
        backend.base_count(layout).map_err(Error::Device)?;
        if layout.records!=0 {
            backend.joint_count(layout.records,rows).map_err(Error::Device)?;
            backend.check_fault(layout.records).map_err(Error::Device)?;
        }
        backend.scan_counts(rows).map_err(Error::Device)?;
        backend.base_fill(layout).map_err(Error::Device)?;
        if layout.records!=0 {backend.joint_fill(layout.records).map_err(Error::Device)?;}
        backend.validate_csr(rows).map_err(Error::Device)?;
        backend.check_fault(rows).map_err(Error::Device)?;
        let (offset,column)=backend.graph(rows).map_err(Error::Device)?;
        let partition_started=std::time::Instant::now();
        let weight=backend.graph_weights();
        let hit=self.graph_partition.as_ref().map_or(false, |cached|
            cached.cloth==layout.cloth && cached.bodies==layout.bodies
            && cached.offset==offset && cached.column==column
            && cached.weight.as_deref()==weight);
        if !hit {
            let domains=partition_weighted(layout.cloth,layout.bodies,&offset,&column,weight)
                .map_err(Error::Invalid)?;
            self.graph_partition=Some(GraphPartition{cloth:layout.cloth,bodies:layout.bodies,
                offset,column,weight:weight.map(<[u32]>::to_vec),domains});
        }
        self.partition_reused=hit;
        self.partition_ms=partition_started.elapsed().as_secs_f64()*1000.;
        let domains=&self.graph_partition.as_ref().expect("partition available").domains;
        self.domains=(domains.offset.len()-1) as u32;
        if self.domains as usize > u32::MAX as usize/DENSE_STRIDE {
            return Err(Error::Invalid("local factor indexing exceeds u32".into()));
        }
        backend.upload_domains(domains).map_err(Error::Device)?;
        backend.gather(self.domains).map_err(Error::Device)?;
        backend.check_fault(self.domains).map_err(Error::Device)?;
        backend.factor(self.domains).map_err(Error::Device)?;
        backend.check_fault(self.domains).map_err(Error::Device)?;
        self.ready=Some(revision);
        Ok(())
    }

    pub fn apply<B: Backend>(&self, backend: &mut B, current: Revision)
        -> Result<(),Error<B::Error>> {
        if self.ready!=Some(current) {return Err(Error::Invalid("stale reduced Schwarz factors".into()));}
        if self.domains==0 {return Ok(());}
        backend.project_input().map_err(Error::Device)?;
        backend.apply(self.domains).map_err(Error::Device)?;
        backend.project_output().map_err(Error::Device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn keeps_body_pairs_complete_cap_and_joint_edges() {
        let (cloth,bodies)=(35u32,13u32);let rows=(cloth+2*bodies) as usize;
        let mut offset=vec![0];let mut column=Vec::new();
        for r in 0..rows {column.push(r as u32);column.push(((r+7)%rows) as u32);offset.push(column.len() as u32);}
        let d=partition(cloth,bodies,&offset,&column).unwrap();
        assert_eq!(d.members.len(),rows);
        assert!(d.offset.windows(2).all(|p| p[1]-p[0]<=CAP as u32));
        for b in 0..bodies {assert_eq!(d.assignment[(cloth+2*b) as usize],d.assignment[(cloth+2*b+1) as usize]);}
        let mut sorted=d.members.clone();sorted.sort_unstable();assert_eq!(sorted,(0..rows as u32).collect::<Vec<_>>());
        assert_eq!(d.assignment,partition(cloth,bodies,&offset,&column).unwrap().assignment);
    }
    #[test] fn disconnected_and_empty_valid() {
        let d=partition(2,1,&[0,1,2,3,4],&[0,1,2,3]).unwrap();
        assert_eq!(d.offset,vec![0,1,2,4]);
        assert!(partition(0,0,&[0],&[]).unwrap().members.is_empty());
        assert!(partition(1,0,&[0,1],&[3]).is_err());
    }
    #[test] fn numeric_edge_weights_select_strongest_neighbor() {
        let mut offset=vec![0];let mut column=Vec::new();let mut weight=Vec::new();
        for r in 0..18u32 {
            column.push(r);weight.push(1);
            if r==0 {for c in 1..18 {column.push(c);weight.push(if c==17 {100} else {1});}}
            else {column.push(0);weight.push(if r==17 {100} else {1});}
            offset.push(column.len() as u32);
        }
        let structural=partition(18,0,&offset,&column).unwrap();
        let numeric=partition_weighted(18,0,&offset,&column,Some(&weight)).unwrap();
        assert_ne!(structural.assignment[0],structural.assignment[17]);
        assert_eq!(numeric.assignment[0],numeric.assignment[17]);
    }
    #[test] fn rejects_u32_extents() {
        assert!(Layout{cloth:u32::MAX,bodies:0,records:0,source_blocks:0}.checked_extents().is_err());
        assert!(Layout{cloth:1,bodies:0,records:u32::MAX,source_blocks:0}.checked_extents().is_err());
        assert!(Layout{cloth:1,bodies:0,records:0,source_blocks:u64::MAX}.checked_extents().is_err());
    }
    #[derive(Default)] struct Mock {events:Vec<&'static str>,fail:bool,weight:Option<Vec<u32>>}
    impl Backend for Mock {
        type Error=();
        fn allocate(&mut self,_:u32,_:usize,_:u32)->Result<(),()>{self.events.push("allocate");Ok(())}
        fn body_seed(&mut self,_:u32)->Result<(),()>{self.events.push("body_seed");Ok(())}
        fn base_count(&mut self,_:Layout)->Result<(),()>{self.events.push("base_count");Ok(())}
        fn joint_count(&mut self,_:u32,_:u32)->Result<(),()>{self.events.push("joint_count");Ok(())}
        fn check_fault(&mut self,_:u32)->Result<(),()>{self.events.push("fault");if self.fail{Err(())}else{Ok(())}}
        fn scan_counts(&mut self,_:u32)->Result<(),()>{self.events.push("scan");Ok(())}
        fn base_fill(&mut self,_:Layout)->Result<(),()>{self.events.push("base_fill");Ok(())}
        fn joint_fill(&mut self,_:u32)->Result<(),()>{self.events.push("joint_fill");Ok(())}
        fn validate_csr(&mut self,_:u32)->Result<(),()>{self.events.push("validate");Ok(())}
        fn graph(&mut self,_:u32)->Result<(Vec<u32>,Vec<u32>),()>{self.events.push("graph");Ok((vec![0,1],vec![0]))}
        fn graph_weights(&self)->Option<&[u32]>{self.weight.as_deref()}
        fn upload_domains(&mut self,_:&Domains)->Result<(),()>{self.events.push("upload");Ok(())}
        fn gather(&mut self,_:u32)->Result<(),()>{self.events.push("gather");Ok(())}
        fn factor(&mut self,_:u32)->Result<(),()>{self.events.push("factor");Ok(())}
        fn project_input(&mut self)->Result<(),()>{self.events.push("project_input");Ok(())}
        fn apply(&mut self,_:u32)->Result<(),()>{self.events.push("apply");Ok(())}
        fn project_output(&mut self)->Result<(),()>{self.events.push("project_output");Ok(())}
    }
    #[test] fn reuses_only_matching_topology_and_still_refactors() {
        let mut backend=Mock::default();
        let mut state=State::default();
        let layout=Layout{cloth:1,bodies:0,records:0,source_blocks:0};
        let rev=Revision{operator:1,mapping:1,joints:1};
        state.rebuild(&mut backend,layout,rev).unwrap();
        let initial=state.graph_partition.as_ref().unwrap().domains.members.as_ptr();
        backend.events.clear();
        state.rebuild(&mut backend,layout,Revision{operator:2,..rev}).unwrap();
        assert_eq!(state.graph_partition.as_ref().unwrap().domains.members.as_ptr(),initial);
        assert!(backend.events.contains(&"gather"));
        assert!(backend.events.contains(&"factor"));
        backend.weight=Some(vec![2]);
        state.rebuild(&mut backend,layout,Revision{operator:3,..rev}).unwrap();
        assert_eq!(state.graph_partition.as_ref().unwrap().weight.as_deref(),Some(&[2][..]));
    }
    #[test] fn ordering_revision_refusal_and_failed_refresh_invalidation() {
        let mut m=Mock::default();let mut state=State::default();
        let rev=Revision{operator:7,mapping:2,joints:11};
        let layout=Layout{cloth:1,bodies:0,records:1,source_blocks:0};
        state.rebuild(&mut m,layout,rev).unwrap();
        assert_eq!(m.events,vec!["allocate","base_count","joint_count","fault","scan","base_fill","joint_fill","validate","fault","graph","upload","gather","fault","factor","fault"]);
        m.events.clear();state.apply(&mut m,rev).unwrap();
        assert_eq!(m.events,vec!["project_input","apply","project_output"]);
        m.events.clear();assert!(state.apply(&mut m,Revision{joints:12,..rev}).is_err());assert!(m.events.is_empty());
        m.fail=true;assert!(state.rebuild(&mut m,layout,rev).is_err());
        assert!(!m.events.contains(&"scan"));m.events.clear();
        assert!(state.apply(&mut m,rev).is_err());assert!(m.events.is_empty());
    }
}
