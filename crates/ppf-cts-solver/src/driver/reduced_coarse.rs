//! Structural planning only. Numerical basis, Galerkin and PCG work stay on GPU.
//! Every child owns at most two triplets; padded coordinates remain explicit.
use std::collections::BTreeSet;
use super::reduced_model::{partition_weighted, Domains};

#[derive(Debug, Clone)]
pub struct LevelPlan {
    pub fine_rows: u32,
    pub aggregates: Domains,
    /// Canonical full symmetric coarse CSR, two triplets per aggregate.
    pub offset: Vec<u32>,
    pub column: Vec<u32>,
}

impl LevelPlan {
    pub fn rows(&self) -> u32 { 2 * (self.aggregates.offset.len() as u32 - 1) }
    pub fn checked(
        cloth: u32, bodies: u32, offset: &[u32], column: &[u32],
    ) -> Result<Self, String> {
        Self::checked_weighted(cloth,bodies,offset,column,None)
    }
    pub fn checked_weighted(cloth:u32,bodies:u32,offset:&[u32],column:&[u32],weights:Option<&[u32]>) ->Result<Self,String>{
        let aggregates=partition_weighted(cloth,bodies,offset,column,weights)?;
        let count=aggregates.offset.len()-1;
        if count > (u32::MAX/6) as usize {return Err("coarse scalar extent overflow".into());}
        let rows=cloth.checked_add(bodies.checked_mul(2).ok_or("body row overflow")?)
            .ok_or("fine row overflow")?;
        let mut neighbors=vec![BTreeSet::<u32>::new();count];
        for a in 0..count {neighbors[a].insert(a as u32);}
        for r in 0..rows as usize {
            let a=aggregates.assignment[r];
            for &c in &column[offset[r] as usize..offset[r+1] as usize] {
                let b=aggregates.assignment[c as usize];
                neighbors[a as usize].insert(b);
                neighbors[b as usize].insert(a);
            }
        }
        let mut coarse_offset=vec![0];
        let mut coarse_column=Vec::new();
        for a in 0..count {
            for _ in 0..2 {
                for &b in &neighbors[a] {
                    coarse_column.push(2*b);
                    coarse_column.push(2*b+1);
                }
                if coarse_column.len() > (u32::MAX/9) as usize {
                    return Err("coarse value extent overflow".into());
                }
                coarse_offset.push(coarse_column.len() as u32);
            }
        }
        Ok(Self{fine_rows:rows,aggregates,offset:coarse_offset,column:coarse_column})
    }
    /// A further level consumes actual Galerkin CSR. Treat every child pair as
    /// indivisible and carry child R transforms through rmc_parent_modes.
    pub fn next(&self) -> Result<Self,String> {
        Self::checked(0,self.rows()/2,&self.offset,&self.column)
    }
}

/// Bounded hierarchy; stop when coarsening makes no structural reduction.
/// Not a claim that more levels necessarily improve wall time.
pub fn hierarchy(cloth:u32,bodies:u32,offset:&[u32],column:&[u32],max_levels:u32)
    -> Result<Vec<LevelPlan>,String> {
    hierarchy_weighted(cloth,bodies,offset,column,max_levels,None)
}
pub fn hierarchy_weighted(cloth:u32,bodies:u32,offset:&[u32],column:&[u32],max_levels:u32,weights:Option<&[u32]>)
    ->Result<Vec<LevelPlan>,String>{
    let mut levels=Vec::new();
    if max_levels==0 {return Ok(levels);}
    let first=LevelPlan::checked_weighted(cloth,bodies,offset,column,weights)?;
    if first.rows()>=first.fine_rows {return Ok(levels);}
    levels.push(first);
    while levels.len() < max_levels as usize {
        let next=levels.last().unwrap().next()?;
        if next.rows()>=next.fine_rows {break;}
        levels.push(next);
    }
    Ok(levels)
}
