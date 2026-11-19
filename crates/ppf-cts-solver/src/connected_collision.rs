//! Host-only connected-body CSR construction. No physics or shared ABI changes.
use crate::cvecvec::CVecVec;
use std::mem::size_of;
pub const FEATURE: &str = "bone_connected_collision_v1";
pub const SEMANTIC_VERSION: u32 = 1;
pub const TAG: u32 = 0x8000_0000;
#[derive(Debug,Clone,serde::Serialize,serde::Deserialize,PartialEq,Eq)]
pub struct Costs { pub exact_nnz:usize, pub final_allocated_bytes:usize, pub device_copy_bytes:usize, pub row_scratch_bytes:usize, pub owner_lookup_bytes:usize, pub cross_product_entries_upper_bound:u64 }
pub type Result<T> = std::result::Result<T, String>;
fn bad(code: &str, detail: &str) -> String { format!("BONE_CONNECTED_{code}: {detail}") }
#[derive(Debug, Clone)]
pub struct Owner { pub uuid: String, pub object: u32, pub members: Vec<u32> }
#[derive(Debug, Clone)]
pub struct Proof { pub owners: Vec<Owner>, pub pairs: Vec<(usize,usize)>, pub input_sha256: String, pub pair_sha256: String, pub owner_sha256: String }
/// Allocation fault injection is used by host tests. Production always uses None.
#[derive(Default)]
pub struct Allocation { pub fail_after: Option<usize>, pub calls: usize }
impl Allocation {
    pub fn reserve<T>(&mut self, v: &mut Vec<T>, additional: usize) -> Result<()> {
        let requested=v.len().checked_add(additional).ok_or_else(||bad("CAPACITY","length overflow"))?;
        checked_span::<T>(requested)?;
        if requested <= v.capacity() { return Ok(()); }
        if self.fail_after==Some(self.calls) { return Err(bad("ALLOCATION","injected reserve failure")); }
        self.calls+=1;
        v.try_reserve_exact(additional).map_err(|e|bad("ALLOCATION",&format!("requested elements={requested}, bytes={}, {e}",requested*size_of::<T>())))?;
        checked_span::<T>(v.capacity())?;
        Ok(())
    }
}
pub fn checked_span<T>(count: usize) -> Result<usize> {
    let bytes=count.checked_mul(size_of::<T>()).ok_or_else(||bad("CAPACITY","byte count overflow"))?;
    if bytes>isize::MAX as usize {return Err(bad("CAPACITY","allocation span exceeds isize"));} Ok(bytes)
}
pub fn checked_layout(n: usize, s: usize, nnz: usize) -> Result<(usize,usize)> {
    if n>=TAG as usize || s>=TAG as usize {return Err(bad("INDEX_DOMAIN","dynamic/static pool exceeds tag boundary"));}
    let rows=n.checked_add(1).ok_or_else(||bad("CAPACITY","rows+1 overflow"))?;
    u32::try_from(rows).map_err(|_|bad("CAPACITY","offset count exceeds u32"))?;
    u32::try_from(nnz).map_err(|_|bad("CAPACITY","nnz exceeds u32 (independent of vertex tag)"))?;
    Ok((checked_span::<u32>(nnz)?,checked_span::<u32>(rows)?))
}
/// Verify *equality* to an entire owner partition, not merely a support subset.
pub fn validate_owners(owners: &[Owner], object: &[u32], rigid: &[u32], fem: &[bool], supports: &[Option<[u32;4]>], simplices: &[Vec<u32>]) -> Result<()> {
    if rigid.len()!=object.len() || owners.len()!=fem.len() || owners.len()!=supports.len() {return Err(bad("OWNERSHIP","domain lengths disagree"));}
    for (i,o) in owners.iter().enumerate() {
        if o.members.is_empty() || o.members.windows(2).any(|w|w[0]>=w[1]) {return Err(bad("OWNERSHIP","members must be nonempty sorted unique"));}
        if owners[..i].iter().any(|a|a.uuid==o.uuid || a.object==o.object) {return Err(bad("OWNERSHIP","duplicate UUID or aliased object owner"));}
        for &v in &o.members {
            if object.get(v as usize)!=Some(&o.object) || (fem[i] && rigid.get(v as usize)!=Some(&0)) {return Err(bad("OWNERSHIP","out-of-range, foreign owner or mixed rigid FEM member"));}
        }
        if object.iter().filter(|&&x|x==o.object).count()!=o.members.len() {return Err(bad("OWNERSHIP","members omit part of the complete object owner set"));}
        if let Some(s)=supports[i] {if s.iter().any(|v|o.members.binary_search(v).is_err()) {return Err(bad("OWNERSHIP","support outside full owner set"));}}
    }
    for simplex in simplices {validate_simplex(owners,object,simplex)?;}
    Ok(())
}
pub fn validate_simplex(owners:&[Owner],object:&[u32],simplex:&[u32])->Result<()> {
    if simplex.iter().any(|v|*v as usize>=object.len()) {return Err(bad("OWNERSHIP","simplex vertex outside dynamic pool"));}
    if simplex.iter().any(|&v|owners.iter().any(|o|o.object==object[v as usize])) && simplex.windows(2).any(|w|object[w[0] as usize]!=object[w[1] as usize]) {return Err(bad("OWNERSHIP","cross-owner collision simplex"));}Ok(())
}
/// Exact two-pass union; scratch is one row, not a cross-product COO or Vec<Vec>.
/// `base` is the original sealed flat combined-pool link list (not loaded derived CSR).
pub fn build(n: usize, s: usize, base: &[u32], owners: &[Owner], pairs: &[(usize,usize)], alloc: &mut Allocation) -> Result<CVecVec<u32>> {Ok(build_reported(n,s,base,owners,pairs,alloc)?.0)}
pub fn build_reported(n: usize, s: usize, base: &[u32], owners: &[Owner], pairs: &[(usize,usize)], alloc: &mut Allocation) -> Result<(CVecVec<u32>,Costs)> {
    checked_layout(n,s,0)?;
    let pool=n.checked_add(s).ok_or_else(||bad("CAPACITY","combined pool overflow"))?;
    if base.len()%2!=0 {return Err(bad("BASE_LINK","odd base pair list"));}
    for p in base.chunks_exact(2) {
        if p[0] as usize>=pool || p[1] as usize>=pool || p[0]==p[1] || (p[0] as usize>=n && p[1] as usize>=n) {return Err(bad("BASE_LINK","invalid original combined-pool pair"));}
    }
    for &(a,b) in pairs {
        if a>=owners.len() || b>=owners.len() || a>=b {return Err(bad("PAIR","canonical distinct owner indices required"));}
    }
    if pairs.windows(2).any(|w|w[0]>=w[1]) {return Err(bad("PAIR","pairs must be sorted unique"));}
    let mut membership=Vec::new();alloc.reserve(&mut membership,n)?;membership.resize(n,usize::MAX);
    for (i,o) in owners.iter().enumerate() { for &v in &o.members {
        if v as usize>=n || membership[v as usize]!=usize::MAX {return Err(bad("OWNERSHIP","overlapping or out-of-range endpoint members"));}
        membership[v as usize]=i;
    }}
    let mut scratch=Vec::<u32>::new();
    let fill=|row:usize,scratch:&mut Vec<u32>,alloc:&mut Allocation| -> Result<()> {
        scratch.clear();
        let mut append=|values:&[u32]| -> Result<()> { alloc.reserve(scratch,values.len())?;scratch.extend_from_slice(values);Ok(()) };
        for p in base.chunks_exact(2) {
            let other=if p[0] as usize==row {Some(p[1])} else if p[1] as usize==row {Some(p[0])} else {None};
            if let Some(v)=other {let v=if (v as usize)<n {v} else {(v as usize-n) as u32|TAG};append(&[v])?;}
        }
        let owner=membership[row];
        for &(a,b) in pairs {if owner==a {append(&owners[b].members)?;} else if owner==b {append(&owners[a].members)?;}}
        scratch.sort_unstable();scratch.dedup();Ok(())
    };
    let mut total=0usize;
    for row in 0..n {fill(row,&mut scratch,alloc)?;total=total.checked_add(scratch.len()).ok_or_else(||bad("CAPACITY","exact union nnz overflow"))?;u32::try_from(total).map_err(|_|bad("CAPACITY","exact union nnz exceeds u32"))?;}
    checked_layout(n,s,total)?;
    let mut data=Vec::<u32>::new();alloc.reserve(&mut data,total)?;
    let mut offset=Vec::<u32>::new();alloc.reserve(&mut offset,n+1)?;offset.push(0);
    for row in 0..n {fill(row,&mut scratch,alloc)?;data.extend_from_slice(&scratch);offset.push(u32::try_from(data.len()).map_err(|_|bad("CAPACITY","prefix exceeds u32"))?);}
    if data.len()!=total {return Err(bad("CAPACITY","union changed between count and fill"));}
    let (db,ob)=checked_layout(n,s,total)?;
    let final_allocated_bytes=checked_span::<u32>(data.capacity())?.checked_add(checked_span::<u32>(offset.capacity())?).ok_or_else(||bad("CAPACITY","final allocation bytes overflow"))?;
    let mut upper=0u64;for &(a,b) in pairs {upper=upper.checked_add((owners[a].members.len() as u64).checked_mul(owners[b].members.len() as u64).and_then(|x|x.checked_mul(2)).ok_or_else(||bad("CAPACITY","pair upper bound overflow"))?).ok_or_else(||bad("CAPACITY","pair upper bound overflow"))?;}
    let costs=Costs{exact_nnz:total,final_allocated_bytes,device_copy_bytes:db.checked_add(ob).ok_or_else(||bad("CAPACITY","device bytes overflow"))?,row_scratch_bytes:checked_span::<u32>(scratch.capacity())?,owner_lookup_bytes:checked_span::<usize>(membership.capacity())?,cross_product_entries_upper_bound:upper};
    Ok((into_cvec(data,offset)?,costs))
}
/// Keeps the *actual* allocation capacities used by Drop. No narrowing cast before validation.
fn into_cvec(mut data:Vec<u32>,mut offset:Vec<u32>) -> Result<CVecVec<u32>> {
    if offset.is_empty() {return Err(bad("CAPACITY","missing zero offset"));}
    let nnz=u32::try_from(data.len()).map_err(|_|bad("CAPACITY","nnz"))?;
    let size=u32::try_from(offset.len()-1).map_err(|_|bad("CAPACITY","rows"))?;
    let nnz_allocated=u32::try_from(data.capacity()).map_err(|_|bad("CAPACITY","actual data allocation count exceeds u32"))?;
    let offset_allocated=u32::try_from(offset.capacity()).map_err(|_|bad("CAPACITY","actual offset allocation count exceeds u32"))?;
    checked_span::<u32>(data.capacity())?;checked_span::<u32>(offset.capacity())?;
    let dp=data.as_mut_ptr();let op=offset.as_mut_ptr();std::mem::forget(data);std::mem::forget(offset);
    Ok(CVecVec{data:dp,offset:op,size,nnz,nnz_allocated,offset_allocated})
}
pub fn slices(table:&CVecVec<u32>) -> Result<(&[u32],&[u32])> {
    if table.size==0 && table.nnz==0 {return Ok((&[],&[]));}
    if table.offset.is_null() || (table.offset_allocated as usize)<(table.size as usize+1) || table.nnz>table.nnz_allocated || (table.nnz>0 && table.data.is_null()) {return Err(bad("CSR","invalid pointer/allocation metadata"));}
    let offsets=unsafe{std::slice::from_raw_parts(table.offset,table.size as usize+1)};
    if offsets.first()!=Some(&0) || offsets.last()!=Some(&table.nnz) || offsets.windows(2).any(|w|w[0]>w[1]) {return Err(bad("CSR","invalid prefix"));}
    let data=if table.nnz==0 {&[]} else {unsafe{std::slice::from_raw_parts(table.data,table.nnz as usize)}};
    Ok((data,offsets))
}
pub fn digest(table:&CVecVec<u32>) -> Result<String> {
    use sha2::{Digest,Sha256};let (d,o)=slices(table)?;let mut h=Sha256::new();h.update(b"ppf.bone_connected.csr/v1\n");h.update(table.size.to_le_bytes());h.update(table.nnz.to_le_bytes());for x in o.iter().chain(d){h.update(x.to_le_bytes());}Ok(format!("{:x}",h.finalize()))
}

/// A deliberately separate host-only command; ordinary solver --path is not reused.
/// Strict mode rejects every extra argument, including normal solver options.
pub const STRICT_INTAKE_FLAG:&str="--bone-connected-collision-strict-intake-json";
pub fn parse_host_intake_args(args:&[String])->Result<Option<(std::path::PathBuf,i32)>> {
    if !args.iter().any(|a|a==STRICT_INTAKE_FLAG) {return Ok(None);}
    if args.first().map(String::as_str)!=Some(STRICT_INTAKE_FLAG) || !matches!(args.len(),2|4) {
        return Err(bad("HOST_USAGE","expected --bone-connected-collision-strict-intake-json SESSION [--load I32] only"));
    }
    if args[1].is_empty() || args[1].starts_with("--") {return Err(bad("HOST_USAGE","a session directory is required"));}
    let load=if args.len()==4 {
        if args[2]!="--load" {return Err(bad("HOST_USAGE","only optional --load I32 is accepted"));}
        args[3].parse::<i32>().map_err(|_|bad("HOST_USAGE","load must be an i32"))?
    } else {0};
    Ok(Some((std::path::PathBuf::from(&args[1]),load)))
}
