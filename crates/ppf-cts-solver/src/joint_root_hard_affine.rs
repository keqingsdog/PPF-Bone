//! Host contract for ONE controlled source root plus existing single D6.
//! Used only by the sealed V7 one-source-root principal runtime branch.
use crate::joint_hard_bridge::{FrozenReducedOperator,PhysicalRow};
use crate::joint_hard_schur::Error;

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Epoch {pub operator:u64,pub root_transaction:u64,pub control_mask:u64}

/// A backend must implement the principal inverse with Q on BOTH sides of
/// every same frozen apply. It cannot solve full K and zero the result later.
/// Intended GPU implementation: existing solve_rigid_impl ReducedRhs with
/// roots:Some(same_frozen_view), zero seed, no particular/gradient/export.
pub trait PrincipalBackend {
    fn epoch(&self)->Epoch;
    fn apply_literal(&mut self,x:&[f64])->Result<Vec<f64>,Error>;
    fn inverse_principal(&mut self,rhs:&[f64],known:&[bool],epoch:Epoch)->Result<Vec<f64>,Error>;
    fn audit_literal_gram(&mut self,b:&[Vec<f64>],y:&[Vec<f64>],g:&[Vec<f64>])->Result<(),Error>;
}

pub struct RootAffine {
    pub known:Vec<bool>,pub particular:Vec<f64>,pub epoch:Epoch,
}
impl RootAffine {
    /// Source/root identity is separately sealed by the V7 topology gate.
    /// `particular` is read back from the real native prescribed Newton slots,
    /// not resampled/reconstructed from author doubles here.
    pub fn one_root(dim:usize,body_base:usize,bodies:usize,body:usize,
                    particular:&[f64],epoch:Epoch)->Result<Self,Error>{
        if particular.len()!=dim||body>=bodies||body_base.checked_add(6*bodies)!=Some(dim)
            ||particular.iter().any(|x|!x.is_finite()){return Err(Error::InvalidInput);}
        let mut known=vec![false;dim];for k in body_base+6*body..body_base+6*body+6{known[k]=true;}
        if particular.iter().zip(&known).any(|(&v,&k)|k&&v!=v as f32 as f64){return Err(Error::InvalidInput);}
        let p=particular.iter().zip(&known).map(|(&v,&known)|if known{v}else{0.}).collect();
        Ok(Self{known,particular:p,epoch})
    }
    pub fn project(&self,x:&[f64])->Result<Vec<f64>,Error>{
        if x.len()!=self.known.len()||x.iter().any(|v|!v.is_finite()){return Err(Error::InvalidInput);}
        Ok(x.iter().zip(&self.known).map(|(&v,&known)|if known{0.}else{v}).collect())
    }
    pub fn enforce_prescribed(&self,u:&[f64])->Result<(),Error>{
        if u.len()!=self.known.len()||u.iter().any(|v|!v.is_finite())
            ||u.iter().zip(&self.particular).zip(&self.known).any(|((&u,&p),&known)|known&&u.to_bits()!=p.to_bits()){return Err(Error::FinalViolation);}Ok(())
    }
    /// Build Q(f-Kp), retaining K_FK*p coupling into cloth/free rigid rows.
    pub fn homogeneous_rhs<B:PrincipalBackend>(&self,backend:&mut B,force:&[f64])->Result<Vec<f64>,Error>{
        self.check_epoch(backend)?;let kp=backend.apply_literal(&self.particular)?;self.check_epoch(backend)?;
        if kp.len()!=force.len()||force.len()!=self.known.len(){return Err(Error::InvalidInput);}
        self.project(&force.iter().zip(kp).map(|(f,k)|f-k).collect::<Vec<_>>())
    }
    pub fn affine_from_homogeneous(&self,v:&[f64])->Result<Vec<f64>,Error>{
        let q=self.project(v)?;Ok(q.iter().zip(&self.particular).zip(&self.known).map(|((v,p),known)|if *known{*p}else{v+p}).collect())
    }
    /// Reactions on prescribed slots are expected, not free-equilibrium errors.
    /// Sign is reduced Newton covector convention; physical restoring sign is
    /// opposite because PPF applies x_trial=x-u.
    pub fn root_reaction<B:PrincipalBackend>(&self,backend:&mut B,u:&[f64],force:&[f64],hard:&[f64])->Result<Vec<f64>,Error>{
        self.enforce_prescribed(u)?;self.check_epoch(backend)?;let ku=backend.apply_literal(u)?;self.check_epoch(backend)?;
        if ku.len()!=force.len()||ku.len()!=hard.len()||ku.len()!=self.known.len(){return Err(Error::InvalidInput);}
        let out:Vec<_>=(0..ku.len()).map(|k|if self.known[k]{ku[k]-force[k]-hard[k]}else{0.}).collect();
        if out.iter().any(|v|!v.is_finite()){return Err(Error::Nonfinite);}Ok(out)
    }
    fn check_epoch<B:PrincipalBackend>(&self,backend:&B)->Result<(),Error>{
        if backend.epoch()!=self.epoch{return Err(Error::FrozenOperatorChanged);}Ok(())
    }
}

pub struct Homogeneous<'a,B> {pub root:&'a RootAffine,pub backend:&'a mut B}
impl<B:PrincipalBackend> FrozenReducedOperator for Homogeneous<'_,B>{
    fn stamp(&self)->u64{self.root.epoch.operator}
    fn apply(&mut self,x:&[f64])->Result<Vec<f64>,Error>{
        self.root.check_epoch(self.backend)?;
        // Input must be homogeneous. Silently projecting a bad vector would
        // hide a leaked prescribed slot in an inverse response.
        if x.len()!=self.root.known.len()||x.iter().zip(&self.root.known).any(|(&v,&known)|known&&v!=0.){return Err(Error::InverseFailed);}
        let out=self.backend.apply_literal(x)?;self.root.check_epoch(self.backend)?;self.root.project(&out)
    }
    fn inverse(&mut self,rhs:&[f64])->Result<Vec<f64>,Error>{
        self.root.check_epoch(self.backend)?;
        if rhs.len()!=self.root.known.len()||rhs.iter().zip(&self.root.known).any(|(&v,&known)|known&&v!=0.){return Err(Error::InverseFailed);}
        let y=self.backend.inverse_principal(rhs,&self.root.known,self.root.epoch)?;self.root.check_epoch(self.backend)?;
        if y.len()!=rhs.len()||y.iter().zip(&self.root.known).any(|(&v,&known)|!v.is_finite()||(known&&v!=0.)){return Err(Error::InverseFailed);}Ok(y)
    }
    fn audit_literal_gram(&mut self,b:&[Vec<f64>],y:&[Vec<f64>],g:&[Vec<f64>])->Result<(),Error>{
        self.root.check_epoch(self.backend)?;self.backend.audit_literal_gram(b,y,g)?;self.root.check_epoch(self.backend)
    }
}

/// Full measured B and coordinate retain root dependence. The existing bridge
/// uses this full B*u0 in the constraint RHS, and Q*B for inverse columns.
pub fn correct<B:PrincipalBackend>(root:&RootAffine,backend:&mut B,rows:&[PhysicalRow],u0:&[f64])->Result<crate::joint_hard_bridge::Correction,Error>{
    root.enforce_prescribed(u0)?;
    let mut homogeneous=Homogeneous{root,backend};
    let mut out=crate::joint_hard_bridge::constrain_literal(&mut homogeneous,rows,u0,&root.known)?;
    // Homogeneous columns have exact known zero. Preserve prescribed bytes
    // even when adding +0 would change a prescribed negative-zero sign.
    for k in 0..out.increment.len(){if root.known[k]{if out.increment[k]!=root.particular[k]{return Err(Error::FinalViolation);}out.increment[k]=root.particular[k];}}
    root.enforce_prescribed(&out.increment)?;Ok(out)
}
