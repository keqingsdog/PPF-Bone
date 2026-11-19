//! Symbolic Galerkin contribution lists. Numeric P, rank and matrix are never cached.
//! Keys contain all derived row/slot/support/coarse address structure. Original
//! vertex/body mapping is not read by this plan; changes enter through these keys
//! or through current numeric handles. In particular no Newton epoch is a key.
use super::reduced_coarse::LevelPlan;

#[derive(Default)]
pub struct Plan {
    fine_offset: Vec<u32>, fine_column: Vec<u32>, owner: Vec<u32>,
    member_offset: Vec<u32>, members: Vec<u32>, coarse_offset: Vec<u32>, coarse_column: Vec<u32>,
    pub pair_offsets: Vec<u32>, pub reverse: Vec<u32>, pub block_pair: Vec<u32>,
    // Each record: fine row, fine CSR slot, left P support slot, right P support slot.
    // Current partition adapter uses row as P support slot. A future multi-support
    // producer can emit multiple records/edge and distinct P slots without changing
    // the numeric representation; its own complete support structure must enter key.
    pub records: Vec<u32>,
}
fn alloc(n:usize)->Option<Vec<u32>> {let mut v=Vec::new();v.try_reserve_exact(n).ok()?;v.resize(n,0);Some(v)}
fn copy(v:&[u32])->Option<Vec<u32>> {let mut r=Vec::new();r.try_reserve_exact(v.len()).ok()?;r.extend_from_slice(v);Some(r)}
impl Plan {
    pub fn matches(&self,fo:&[u32],fc:&[u32],p:&LevelPlan)->bool {
        self.fine_offset==fo && self.fine_column==fc && self.owner==p.aggregates.assignment &&
        self.member_offset==p.aggregates.offset && self.members==p.aggregates.members &&
        self.coarse_offset==p.offset && self.coarse_column==p.column
    }
    pub fn host_bytes(&self)->usize {
        [&self.fine_offset,&self.fine_column,&self.owner,&self.member_offset,&self.members,
         &self.coarse_offset,&self.coarse_column,&self.pair_offsets,&self.reverse,&self.block_pair,&self.records]
        .iter().fold(0usize,|n,v|n.saturating_add(v.capacity().saturating_mul(4)))
    }
}
// Distinguish optional resource refusal from a malformed structural plan.
pub enum Outcome { Ready(Plan), Resource, Invalid(&'static str) }
fn pair(p:&LevelPlan,a:usize,b:u32)->Option<usize> {
    let start=*p.offset.get(2*a)? as usize;let end=*p.offset.get(2*a+1)? as usize;
    let s=p.column.get(start..end)?; if s.len()%2!=0 {return None;}
    let mut l=0;let mut h=s.len()/2;let target=b.checked_mul(2)?;
    while l<h {let m=(l+h)/2;if s[2*m]<target {l=m+1}else{h=m}}
    if l<s.len()/2 && s[2*l]==target && s[2*l+1]==target+1 {Some(start/4+l)} else {None}
}
/// Logical peak estimate includes new host key/plan, counts/cursor scratch,
/// device copy, and old buffers during growth; caller adds all live old plans.
pub fn peak_upper(fo:&[u32],fc:&[u32],p:&LevelPlan)->Option<usize> {
    let pairs=p.column.len().checked_div(4)?;let edges=fc.len();
    let key=fo.len().checked_add(fc.len())?.checked_add(p.aggregates.assignment.len())?
        .checked_add(p.aggregates.offset.len())?.checked_add(p.aggregates.members.len())?
        .checked_add(p.offset.len())?.checked_add(p.column.len())?;
    let words=key.checked_add(pairs.checked_add(1)?)?.checked_add(pairs)?
        .checked_add(p.column.len())?.checked_add(edges.checked_mul(4)?)?;
    let scratch=pairs.checked_mul(2)?.checked_add(1)?;
    // host candidate plus two device-size allowances during realloc, and scratch.
    words.checked_mul(3)?.checked_add(scratch)?.checked_mul(4)
}
pub fn build(fo:&[u32],fc:&[u32],p:&LevelPlan)->Outcome {
    let rows=p.fine_rows as usize;let nodes=p.aggregates.offset.len().saturating_sub(1);
    if fo.len()!=rows+1 || fo.first()!=Some(&0) || fo.last().copied()!=Some(fc.len() as u32) ||
       fo.windows(2).any(|x|x[0]>x[1]) || p.aggregates.assignment.len()!=rows ||
       p.aggregates.members.len()!=rows || p.aggregates.offset.first()!=Some(&0) ||
       p.aggregates.offset.last().copied()!=Some(rows as u32) || p.offset.len()!=2*nodes+1 ||
       p.offset.first()!=Some(&0) || p.offset.last().copied()!=Some(p.column.len() as u32) || p.column.len()%4!=0 {
        return Outcome::Invalid("invalid Galerkin structure extents");
    }
    let pairs=p.column.len()/4;
    let Some(mut counts)=alloc(pairs) else{return Outcome::Resource};
    let Some(mut reverse)=alloc(pairs) else{return Outcome::Resource};
    let Some(mut block_pair)=alloc(p.column.len()) else{return Outcome::Resource};
    for a in 0..nodes {
        let lo=p.offset[2*a] as usize;let hi=p.offset[2*a+1] as usize;
        let lo2=p.offset[2*a+1] as usize;let hi2=p.offset[2*a+2] as usize;
        if hi>p.column.len()||hi2>p.column.len()||hi<lo||hi2<lo2||hi-lo!=hi2-lo2||lo%4!=0 ||
           p.column[lo..hi]!=p.column[lo2..hi2] {return Outcome::Invalid("invalid paired coarse rows");}
        for k in (lo..hi).step_by(2) {
            if k+1>=hi||p.column[k]%2!=0||p.column[k+1]!=p.column[k]+1 ||
                (k>lo && p.column[k-2]>=p.column[k]) {return Outcome::Invalid("invalid coarse neighbor ordering");}
            let b=(p.column[k]/2) as usize;if b>=nodes{return Outcome::Invalid("coarse neighbor out of range");}
            let id=lo/4+(k-lo)/2; let Some(back)=pair(p,b,a as u32) else{return Outcome::Invalid("missing coarse reverse pair")};
            reverse[id]=back as u32;
            for off in 0..2 {block_pair[k+off]=id as u32;block_pair[lo2+k-lo+off]=id as u32;}
        }
        if pair(p,a,a as u32).is_none(){return Outcome::Invalid("missing coarse self pair");}
    }
    for a in 0..nodes {
        let lo=p.aggregates.offset[a] as usize;let hi=p.aggregates.offset[a+1] as usize;
        if lo>hi||hi>rows{return Outcome::Invalid("invalid member offsets");}
        if p.aggregates.members[lo..hi].windows(2).any(|w|w[0]>=w[1]) {return Outcome::Invalid("duplicate or unordered members");}
        for &fr in &p.aggregates.members[lo..hi] {
            let r=fr as usize;if r>=rows||p.aggregates.assignment[r] as usize!=a{return Outcome::Invalid("invalid support membership");}
            for k in fo[r] as usize..fo[r+1] as usize {
                let c=fc[k] as usize;if c>=rows{return Outcome::Invalid("invalid fine column");}
                let Some(id)=pair(p,a,p.aggregates.assignment[c]) else{return Outcome::Invalid("missing contribution pair")};
                let Some(n)=counts[id].checked_add(1) else{return Outcome::Resource};counts[id]=n;
            }
        }
    }
    let Some(mut offsets)=alloc(pairs+1) else{return Outcome::Resource};
    for i in 0..pairs {let Some(v)=offsets[i].checked_add(counts[i]) else{return Outcome::Resource};offsets[i+1]=v;}
    if offsets[pairs] as usize!=fc.len() {return Outcome::Invalid("fine contribution count mismatch");}
    let Some(mut cursor)=copy(&offsets[..pairs]) else{return Outcome::Resource};
    let Some(n)=fc.len().checked_mul(4) else{return Outcome::Resource};let Some(mut records)=alloc(n) else{return Outcome::Resource};
    for a in 0..nodes {for &fr in &p.aggregates.members[p.aggregates.offset[a] as usize..p.aggregates.offset[a+1] as usize] {
        for k in fo[fr as usize] as usize..fo[fr as usize+1] as usize {let c=fc[k];let id=pair(p,a,p.aggregates.assignment[c as usize]).unwrap();
            let pos=4*cursor[id] as usize;records[pos]=fr;records[pos+1]=k as u32;records[pos+2]=fr;records[pos+3]=c;cursor[id]+=1;
        }
    }}
    macro_rules! c {($x:expr)=>{match copy($x){Some(v)=>v,None=>return Outcome::Resource}}}
    Outcome::Ready(Plan{fine_offset:c!(fo),fine_column:c!(fc),owner:c!(&p.aggregates.assignment),
        member_offset:c!(&p.aggregates.offset),members:c!(&p.aggregates.members),coarse_offset:c!(&p.offset),
        coarse_column:c!(&p.column),pair_offsets:offsets,reverse,block_pair,records})
}


