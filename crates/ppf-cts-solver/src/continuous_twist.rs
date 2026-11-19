//! Transactional per-full-joint continuous twist history.
//!
//! This state does not establish that a path is unambiguous. The caller must
//! certify every actual pose segment before advancing the trial branch.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct AcceptedTwist {
    pub scene_uuid: String,
    pub joint_uuids: Vec<String>,
    pub time: f64,
    pub theta: Vec<f64>,
    /// False when migrated from a checkpoint that did not preserve turns.
    pub prior_turns_known: bool,
}

impl AcceptedTwist {
    pub fn validate(&self, scene: &str, joints: &[String], time: f64) -> Result<(), &'static str> {
        if self.scene_uuid != scene || self.joint_uuids != joints || self.theta.len() != joints.len() {
            return Err("continuous twist joint identity mismatch");
        }
        if !time.is_finite() || time < 0.0 || self.time.to_bits() != time.to_bits()
            || self.theta.iter().any(|x| !x.is_finite()) {
            return Err("continuous twist history invalid");
        }
        Ok(())
    }
    pub fn trial(&self) -> TrialTwist { TrialTwist { theta: self.theta.clone() } }
    pub fn commit(&mut self, trial: TrialTwist, time: f64) -> Result<(), &'static str> {
        if trial.theta.len() != self.theta.len() || trial.theta.iter().any(|x| !x.is_finite())
            || !time.is_finite() || time <= self.time {
            return Err("continuous twist commit invalid");
        }
        self.theta = trial.theta;
        self.time = time;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TrialTwist { pub theta: Vec<f64> }
pub(crate) fn net_segment_within_pi(selected:&[bool],start:&[f64],end:&[f64])->bool {
    selected.len()==start.len() && start.len()==end.len() && selected.iter().enumerate().all(|(i,s)| !*s || (start[i].is_finite() && end[i].is_finite() && (end[i]-start[i]).abs()<std::f64::consts::PI-1e-7))
}
impl TrialTwist {
    /// The path leading to each principal phase must already satisfy the
    /// certified strict less-than-pi twist increment bound.
    pub fn advance_certified(&mut self, principal: &[f64]) -> Result<(), &'static str> {
        if principal.len() != self.theta.len() { return Err("continuous twist count mismatch"); }
        let mut next = Vec::with_capacity(principal.len());
        for (&anchor, &phase) in self.theta.iter().zip(principal) {
            if !anchor.is_finite() || !phase.is_finite() || phase.abs() > std::f64::consts::PI + 1e-6 {
                return Err("continuous twist phase invalid");
            }
            let turns = ((anchor - phase) / std::f64::consts::TAU).round();
            let lifted = phase + turns * std::f64::consts::TAU;
            if !lifted.is_finite() || (lifted - anchor).abs() >= std::f64::consts::PI - 1e-7 {
                return Err("continuous twist phase ambiguous");
            }
            next.push(lifted);
        }
        self.theta = next;
        Ok(())
    }
}

/// Project a lifted phase for device evaluation without silently losing turns
/// or allowing accumulated rounding to exceed the angular tolerance.
pub(crate) fn project_device(theta: &[f64]) -> Result<Vec<f32>, &'static str> {
    const MAX_PROJECTION_ERROR_RAD: f64 = 1e-4;
    theta.iter().map(|&value| {
        let projected = value as f32;
        if !value.is_finite() || value.abs() > 512.0 || !projected.is_finite()
            || (f64::from(projected) - value).abs() > MAX_PROJECTION_ERROR_RAD {
            return Err("continuous twist device projection loses precision");
        }
        Ok(projected)
    }).collect()
}


/// Conservative proof for one actual rotation path segment. `angle_a_bound`
/// and `angle_b_bound` must be upper bounds, including fit and root override;
/// sampling just the final poses cannot supply those bounds. Returns false for
/// a singular/uncertifiable segment; callers must subdivide or retry.
pub(crate) fn certified_segment(radius_lower: f64, angle_a_bound: f64, angle_b_bound: f64) -> bool {
    const CUT: f64 = 1e-4;
    const MARGIN: f64 = 1e-5;
    if !radius_lower.is_finite() || !angle_a_bound.is_finite() || !angle_b_bound.is_finite()
        || radius_lower <= 0.0 || radius_lower > 1.0
        || angle_a_bound < 0.0 || angle_b_bound < 0.0 {
        return false;
    }
    let length_upper = (angle_a_bound + angle_b_bound) * 0.5;
    length_upper.is_finite() && radius_lower - length_upper > CUT
        && length_upper < radius_lower * (1.0 - MARGIN)
}

/// Device-side per-full-joint branch context. Authored control rows retain
/// their existing ABI; this context is rebuilt only at safe stage boundaries.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AngularTwistEvalContext {
    pub current_anchor: f32,
    pub reference_anchor: f32,
    pub continuous: u32,
    pub initialized: u32,
}
unsafe impl ppf_cts_compute::Pod for AngularTwistEvalContext {}
const _: () = assert!(std::mem::size_of::<AngularTwistEvalContext>() == 16);

/// Reconstruct the same unweighted x * rest^T covariance used by
/// pdrd_fit_covariance_row from the stored f32 inputs. The returned absolute
/// Frobenius error is a conservative f64 roundoff envelope; GPU endpoint fit
/// error is handled separately through the sampled device quaternions.
pub(crate) fn covariance_from_vertices(positions:&[[f32;3]],rest:&[[f32;3]]) -> Option<([f64;9],f64)> {
    let n=positions.len();
    if n==0 || rest.len()!=n {return None;}
    let u=f64::EPSILON/2.0;
    let k=(8.0*n as f64+32.0)*u;
    if k>=1e-3 {return None;}
    let gamma=k/(1.0-k);
    let origin=positions[0].map(f64::from);
    let mut sy=[0.0_f64;3]; let mut sr=[0.0_f64;3];
    let mut ay=[0.0_f64;3]; let mut ar=[0.0_f64;3];
    let mut products=[0.0_f64;9]; let mut ap=[0.0_f64;9];
    for (p,r) in positions.iter().zip(rest) {
        if p.iter().chain(r.iter()).any(|v|!v.is_finite()) {return None;}
        let y=std::array::from_fn::<_,3,_>(|i|f64::from(p[i])-origin[i]);
        for i in 0..3 {sy[i]+=y[i];sr[i]+=f64::from(r[i]);ay[i]+=y[i].abs();ar[i]+=f64::from(r[i]).abs();}
        for col in 0..3 {for row in 0..3 {let i=row+3*col;let v=y[row]*f64::from(r[col]);products[i]+=v;ap[i]+=v.abs();}}
    }
    let mut c=[0.0_f64;9]; let mut error_squared=0.0_f64;
    for col in 0..3 {for row in 0..3 {
        let i=row+3*col;
        c[i]=products[i]-(sy[row]/n as f64)*sr[col];
        let e=16.0*gamma*(ap[i]+(ay[row]/n as f64)*ar[col]);
        error_squared+=e*e;
    }}
    let norm=c.iter().map(|v|v*v).sum::<f64>().sqrt();
    let error=error_squared.sqrt()+norm*1e-12;
    if !norm.is_finite() || !error.is_finite() {return None;}
    Some((c,error))
}

/// Read-only reconstruction from the same staged topology used by Reduction.
/// Unselected bodies are not inspected and produce None.
pub(crate) fn body_covariances(
    positions:&[f32], vert_list:&[u32], rest_centered:&[f32],
    bodies:&[crate::data::PdrdBodyProp], selected:&[bool],
) -> Option<Vec<Option<([f64;9],f64)>>> {
    if bodies.len()!=selected.len() || positions.len()%3!=0 || rest_centered.len()%3!=0 {return None;}
    let mut out=Vec::with_capacity(bodies.len());
    for (body,&use_body) in bodies.iter().zip(selected) {
        if !use_body {out.push(None);continue;}
        let start=body.vertex_start as usize;
        let end=start.checked_add(body.vertex_count as usize)?;
        if end>vert_list.len() || end>rest_centered.len()/3 {return None;}
        let mut xyz=Vec::with_capacity(end-start);
        let mut rest=Vec::with_capacity(end-start);
        for j in start..end {
            let v=vert_list[j] as usize;
            if v>=positions.len()/3 {return None;}
            xyz.push([positions[v*3],positions[v*3+1],positions[v*3+2]]);
            rest.push([rest_centered[j*3],rest_centered[j*3+1],rest_centered[j*3+2]]);
        }
        out.push(Some(covariance_from_vertices(&xyz,&rest)?));
    }
    Some(out)
}

/// Validated proper polar fit with explicit rejection reasons for diagnostics.
pub(crate) fn accurate_fit_rotation(covariance:[f32;9])->Option<[f32;9]>{
    accurate_fit_rotation_details(covariance).ok()
}

pub(crate) fn accurate_fit_rotation_details(covariance:[f32;9])->Result<[f32;9], &'static str>{
    use nalgebra::{Matrix3,Vector3};
    if covariance.iter().any(|x|!x.is_finite()){return Err("nonfinite_input");}
    let c=Matrix3::from_column_slice(&covariance.map(f64::from));let scale=c.norm();
    if !scale.is_finite() || scale<=0.0 || c.determinant()<=0.0{return Err("norm_or_determinant");}
    let mut p=c/scale;let mut converged=false;
    for _ in 0..64 {
        let inv=p.try_inverse().ok_or("newton_inverse")?;p=(p+inv.transpose())*0.5;
        if p.iter().any(|v|!v.is_finite()){return Err("newton_nonfinite");}
        if (p.transpose()*p-Matrix3::identity()).norm()<=1e-12 {converged=true;break;}
    }
    let eq=(p.transpose()*p-Matrix3::identity()).norm();
    if !converged || !eq.is_finite() || p.determinant()<=0.0 || eq>1e-9{return Err("rotation");}
    let h=(p.transpose()*c+c.transpose()*p)*0.5;
    if h.iter().any(|v|!v.is_finite()){return Err("nonfinite_stretch");}
    let mut a=h;let mut v=Matrix3::<f64>::identity();let mut eigen_converged=false;
    for _ in 0..192 {
        let mut pair=(0usize,1usize,a[(0,1)].abs());
        for (i,j) in [(0usize,2usize),(1usize,2usize)] {let x=a[(i,j)].abs();if x>pair.2{pair=(i,j,x);}}
        if !pair.2.is_finite(){return Err("jacobi_nonfinite");}
        if pair.2<=scale*1e-14{eigen_converged=true;break;}
        let (i,j,_)=pair;let off=a[(i,j)];let tau=(a[(j,j)]-a[(i,i)])/(2.0*off);
        if !tau.is_finite(){return Err("jacobi_nonfinite");}
        let t=if tau==0.0{1.0}else{tau.signum()/(tau.abs()+tau.hypot(1.0))};
        let cosine=1.0/(1.0+t*t).sqrt();let sine=t*cosine;
        let ai=a[(i,i)];let aj=a[(j,j)];a[(i,i)]=ai-t*off;a[(j,j)]=aj+t*off;a[(i,j)]=0.0;a[(j,i)]=0.0;
        for k in 0..3 {if k!=i && k!=j {let ki=a[(k,i)];let kj=a[(k,j)];a[(k,i)]=cosine*ki-sine*kj;a[(i,k)]=a[(k,i)];a[(k,j)]=sine*ki+cosine*kj;a[(j,k)]=a[(k,j)];}}
        for k in 0..3 {let ki=v[(k,i)];let kj=v[(k,j)];v[(k,i)]=cosine*ki-sine*kj;v[(k,j)]=sine*ki+cosine*kj;}
    }
    if !eigen_converged || a.iter().chain(v.iter()).any(|x|!x.is_finite()){return Err("jacobi_not_converged");}
    let singular=Vector3::new(a[(0,0)],a[(1,1)],a[(2,2)]);
    if singular.iter().any(|s|!s.is_finite() || *s<=scale*1e-7){return Err("singular_values");}
    let diagonal=Matrix3::from_diagonal(&singular);let reconstructed=p*v*diagonal*v.transpose();
    if reconstructed.iter().any(|x|!x.is_finite()) || (reconstructed-c).norm()>scale*1e-9{return Err("svd_reconstruction");}
    let ev=(v.transpose()*v-Matrix3::identity()).norm();let eh=(h-v*diagonal*v.transpose()).norm();let polar=(c-p*h).norm();
    if !ev.is_finite() || !eh.is_finite() || !polar.is_finite() || eq>=1.0 || ev>=1.0{return Err("bound_nonfinite");}
    let lower=(1.0-eq).sqrt()*((1.0-ev)*singular.min()-eh)-polar;
    if !lower.is_finite() || lower<=scale*1e-7{return Err("singular_lower_bound");}
    if polar>scale*1e-9{return Err("polar_reconstruction");}
    let out:[f32;9]=std::array::from_fn(|i|p.as_slice()[i] as f32);let projected=Matrix3::from_column_slice(&out.map(f64::from));
    if projected.iter().any(|x|!x.is_finite()) || projected.determinant()<=0.0 || (projected.transpose()*projected-Matrix3::identity()).norm()>2e-6 || (projected-p).norm()>2e-6{return Err("f32_projection");}
    Ok(out)
}

/// Conservative polar-path angular length, conditional on the caller supplying
/// covariance/rotation arrays with a conservative absolute error envelope.
/// Column-major matrices match PDRD. None means the segment is uncertified.
#[derive(Clone,Copy)]
pub(crate) struct PolarSegmentParts { pub start:f64,pub path:f64,pub end:f64,pub polar0:[f64;9],pub polar1:[f64;9] }

/// Independent, condition-aware polar-path bound. Input errors are absolute
/// Frobenius envelopes for both covariance endpoints. None leaves the caller
/// free to use only its independently certified absolute bound.
fn relative_polar_path_bound(c0:&nalgebra::Matrix3<f64>,c1:&nalgebra::Matrix3<f64>,error:f64)->Option<f64>{
    use nalgebra::Matrix3;
    const K:f64=1e-10;
    let normal=|x:f64|x.is_finite() && (x==0.0 || x.is_normal());
    let norm=|m:&Matrix3<f64>|m.iter().fold(0.0_f64,|acc,x|acc.hypot(*x));
    if !normal(error) || error<0.0 || c0.iter().chain(c1.iter()).any(|v|!normal(*v)){return None;}
    let inverse=c0.try_inverse()?;
    if inverse.iter().any(|v|!normal(*v)){return None;}
    let delta=c1-c0;
    if delta.iter().any(|v|!normal(*v)){return None;}
    let product=c0*inverse;
    let scaled=delta*inverse;
    if product.iter().chain(scaled.iter()).any(|v|!normal(*v)){return None;}
    let n0=norm(c0);let n1=norm(c1);let nd=norm(&delta);let ni=norm(&inverse);
    if [n0,n1,nd,ni].iter().any(|v|!normal(*v)) || ni<=0.0{return None;}
    // K covers f64 subtraction, matrix products and norms conservatively;
    // the independently supplied error covers the two original covariances.
    let sum=n0+n1;let two_error=2.0*error;let c0_inverse=n0*ni;
    let delta_inverse=nd*ni;let input_inverse=error*ni;
    if [sum,two_error,c0_inverse,delta_inverse,input_inverse].iter().any(|v|!normal(*v)){return None;}
    let delta_error=two_error+K*sum;
    let delta_error_inverse=delta_error*ni;
    if !normal(delta_error_inverse){return None;}
    let rho=norm(&(Matrix3::identity()-product))+input_inverse+K*(1.0+c0_inverse);
    let beta=norm(&scaled)+delta_error_inverse+K*(1.0+delta_inverse);
    if [delta_error,rho,beta].iter().any(|v|!normal(*v)) || rho<0.0 || rho>=1.0 || beta<0.0{return None;}
    let raw=beta/(1.0-rho);
    let a=raw+K*(1.0+raw);
    if !normal(raw) || !normal(a) || a<0.0 || a>=1.0{return None;}
    let bound=-(-a).ln_1p();let guarded=bound+K*(1.0+bound);
    if normal(bound) && normal(guarded) {Some(guarded)} else {None}
}
pub(crate) fn polar_segment_parts(c0: [f64;9], c1: [f64;9], q0: [f64;4], q1: [f64;4], covariance_error: f64) -> Option<PolarSegmentParts> {
    use nalgebra::{Matrix3, Quaternion, UnitQuaternion};
    if !covariance_error.is_finite() || covariance_error<0.0 ||
        c0.iter().chain(c1.iter()).chain(q0.iter()).chain(q1.iter()).any(|x|!x.is_finite()) { return None; }
    let c0=Matrix3::from_column_slice(&c0);
    let c1=Matrix3::from_column_slice(&c1);
    // q is the quaternion emitted by the production device chart, not a
    // second host matrix-to-quaternion conversion. Conversion is explicitly
    // normalized in f64 and a dimensionless f32 margin is added below.
    let endpoint=|c:&Matrix3<f64>,q:[f64;4]| -> Option<(f64,f64)> {
        let raw=Quaternion::new(q[0],q[1],q[2],q[3]);
        let norm=raw.norm();
        if !norm.is_finite() || (norm-1.0).abs()>1e-4 {return None;}
        let rotation=UnitQuaternion::new_normalize(raw).to_rotation_matrix();
        let qm=rotation.matrix();
        let scale=c.norm();
        if !scale.is_finite() || scale<=0.0 {return None;}
        let guard=scale*1e-10+covariance_error;
        let h=(qm.transpose()*c+c.transpose()*qm)*0.5;
        // Positivity is required: an inverse norm alone would accept a
        // reflected or negative-definite matrix with the wrong polar factor.
        let l11=h[(0,0)];
        if l11<=guard {return None;}
        let l21=h[(1,0)]/l11.sqrt();
        let l31=h[(2,0)]/l11.sqrt();
        let l22=h[(1,1)]-l21*l21;
        if l22<=guard {return None;}
        let l32=(h[(2,1)]-l31*l21)/l22.sqrt();
        let l33=h[(2,2)]-l31*l31-l32*l32;
        if l33<=guard || c.determinant()<=guard*scale*scale {return None;}
        let inverse=h.try_inverse()?;
        let residual=(Matrix3::identity()-h*inverse).norm();
        let inverse_norm=inverse.norm();
        if !residual.is_finite() || residual>=1.0 || !inverse_norm.is_finite() || inverse_norm<=0.0 {return None;}
        let mu=(1.0-residual)/inverse_norm-guard;
        let delta=(c-qm*h).norm()+guard;
        if mu<=0.0 || delta>=mu {return None;}
        let chord=2.0*delta/(2.0*mu-delta);
        // Frobenius rotation chord to angular bound. The extra 1e-5 rad
        // accounts for device f32 quaternion normalization arithmetic.
        let correction=std::f64::consts::PI*chord/(2.0*2.0_f64.sqrt())+1e-5;
        Some((mu-delta,correction))
    };
    let polar=|c:Matrix3<f64>|->Option<[f64;9]> {
        let svd=c.svd(true,true);
        if svd.singular_values.iter().any(|s|!s.is_finite() || *s<=covariance_error+c.norm()*1e-10) {return None;}
        let p=svd.u?*svd.v_t?;
        if p.determinant()<=0.0 || (p.transpose()*p-Matrix3::identity()).norm()>1e-10 {return None;}
        Some(std::array::from_fn(|i|p.as_slice()[i]))
    };
    let polar0=polar(c0)?;let polar1=polar(c1)?;
    // The SVD polar is an approximation too. Bound its distance to the true
    // polar with the same residual/SPD test, then apply the angular triangle
    // inequality from the actual device quaternion through that approximation.
    // Taking the smaller of two independently conservative bounds is safe.
    let polar_endpoint=|c:&Matrix3<f64>,p:[f64;9]|->Option<(f64,f64)>{
        let qp=UnitQuaternion::from_matrix(&Matrix3::from_column_slice(&p));
        let pquat=qp.quaternion();
        endpoint(c,[pquat.w,pquat.i,pquat.j,pquat.k])
    };
    // Covariance nonsingularity depends on the covariance and its true polar,
    // not on the accuracy of the approximate device fit at this endpoint.
    let (sigma0,polar_error0)=polar_endpoint(&c0,polar0)?;
    let (_,polar_error1)=polar_endpoint(&c1,polar1)?;
    let d=(c1-c0).norm()+2.0*covariance_error;
    let margin=sigma0-d;
    let absolute=if d.is_finite() && margin>0.0 {let v=2.0*d/margin;if v.is_finite(){Some(v)}else{None}}else{None};
    let relative=relative_polar_path_bound(&c0,&c1,covariance_error);
    let path=match (absolute,relative) {(Some(a),Some(b))=>a.min(b),(Some(a),None)=>a,(None,Some(b))=>b,(None,None)=>return None};
    let correction=|c:&Matrix3<f64>,gpu:[f64;4],p:[f64;9],polar_error:f64|->Option<f64>{
        let qp=UnitQuaternion::from_matrix(&Matrix3::from_column_slice(&p));
        let raw=Quaternion::new(gpu[0],gpu[1],gpu[2],gpu[3]);
        let norm=raw.norm();
        if !norm.is_finite() || (norm-1.0).abs()>1e-4 {return None;}
        let qgpu=UnitQuaternion::new_normalize(raw);
        let relative=qp.inverse()*qgpu;let q=relative.quaternion();
        let sin_half=(q.i*q.i+q.j*q.j+q.k*q.k).sqrt();
        let angle=2.0*sin_half.atan2(q.w.abs());
        if !angle.is_finite() || angle>=std::f64::consts::PI-1e-5 {return None;}
        let candidate=angle+polar_error+1e-5;
        if !candidate.is_finite(){return None;}
        Some(endpoint(c,gpu).map_or(candidate,|(_,old)|old.min(candidate)))
    };
    Some(PolarSegmentParts {start:correction(&c0,q0,polar0,polar_error0)?,path,end:correction(&c1,q1,polar1,polar_error1)?,polar0,polar1})
}
pub(crate) fn polar_segment_bound(c0:[f64;9],c1:[f64;9],q0:[f64;4],q1:[f64;4],error:f64)->Option<f64>{
    let p=polar_segment_parts(c0,c1,q0,q1,error)?;Some(p.start+p.path+p.end)
}

#[cfg(test)]
mod polar_tests {
    use super::*;
    #[test]
    fn covariance_translation_and_reflection_guards() {
        let rest=[[-1.,-1.,-1.],[1.,-1.,-1.],[-1.,1.,-1.],[1.,1.,-1.],[-1.,-1.,1.],[1.,-1.,1.],[-1.,1.,1.],[1.,1.,1.]];
        let shifted=rest.map(|p|[p[0]+100.,p[1]-200.,p[2]+50.]);
        let (c,e)=covariance_from_vertices(&shifted,&rest).unwrap();
        assert_eq!(c,[8.,0.,0.,0.,8.,0.,0.,0.,8.]);
        assert!(polar_segment_bound(c,c,[1.,0.,0.,0.],[1.,0.,0.,0.],e).unwrap()<0.001);
        let reflected=[-8.,0.,0.,0.,8.,0.,0.,0.,8.];
        assert!(polar_segment_bound(reflected,reflected,[1.,0.,0.,0.],[1.,0.,0.,0.],e).is_none());
        assert!(covariance_from_vertices(&[[f32::NAN,0.,0.]],&[[0.,0.,0.]]).is_none());
    }
    #[test]
    fn irregular_covariance_matches_direct_centroid_oracle() {
        let p=[[1024.,1026.,1027.],[1027.,1023.,1028.],[1023.,1029.,1022.],[1031.,1024.,1023.],[1021.,1022.,1032.],[1025.,1030.,1025.]];
        let r=[[1.,2.,3.],[-1.,3.,2.],[2.,-2.,1.],[4.,1.,-3.],[-3.,0.,2.],[1.,4.,-2.]];
        let (actual,error)=covariance_from_vertices(&p,&r).unwrap();
        let mean: [f64;3]=std::array::from_fn(|i|p.iter().map(|v|f64::from(v[i])).sum::<f64>()/p.len() as f64);
        for row in 0..3 {for col in 0..3 {
            let expected=p.iter().zip(r.iter()).map(|(x,y)|(f64::from(x[row])-mean[row])*f64::from(y[col])).sum::<f64>();
            assert!((actual[row+3*col]-expected).abs()<=error);
        }}
    }
    #[test]
    fn small_segment_certifies_and_hidden_half_turn_rejects() {
        let c0=[8.,0.,0.,0.,8.,0.,0.,0.,8.];
        let angle=10_f64.to_radians();
        let (s,c)=angle.sin_cos();
        let c1=[8.,0.,0.,0.,8.*c,8.*s,0.,-8.*s,8.*c];
        let q1=[(angle/2.).cos(),(angle/2.).sin(),0.,0.];
        let bound=polar_segment_bound(c0,c1,[1.,0.,0.,0.],q1,1e-10).unwrap();
        assert!(bound>=angle && certified_segment(0.99,bound,0.0));
        assert!(!certified_segment(0.99,bound+std::f64::consts::TAU,0.0));
        let half_turn=[8.,0.,0.,0.,-8.,0.,0.,0.,-8.];
        assert!(polar_segment_bound(c0,half_turn,[1.,0.,0.,0.],[0.,1.,0.,0.],1e-10).is_none());
    }
    #[test]
    fn three_piece_transport_cancels_offsets_crosses_seam_and_rejects_detour() {
        use std::f64::consts::{PI,TAU};
        for anchor in [0.0,2.0*TAU,-2.0*TAU] {
            let mut t=TrialTwist{theta:vec![anchor]};
            for phase in [0.2,-0.1,0.0] {t.advance_certified(&[phase]).unwrap();}
            assert!((t.theta[0]-anchor).abs()<1e-10);
            assert!(net_segment_within_pi(&[true],&[anchor],&t.theta));
        }
        let start=PI-0.05;let mut t=TrialTwist{theta:vec![start]};
        for phase in [PI-0.02,-PI+0.01,-PI+0.04] {t.advance_certified(&[phase]).unwrap();}
        assert!(t.theta[0]>PI && net_segment_within_pi(&[true],&[start],&t.theta));
        let mut detour=TrialTwist{theta:vec![0.0]};
        for phase in [2.5,-1.28,0.0] {detour.advance_certified(&[phase]).unwrap();}
        assert!((detour.theta[0]-TAU).abs()<1e-10);
        assert!(!net_segment_within_pi(&[true],&[0.0],&detour.theta));
    }
    #[test]
    fn captured_m2_three_parts_can_be_certified_without_dropping_corrections() {
        let c00=[0.005247735165628625,0.4151096302173304,3.0533591642092013,-0.861882134737031,-0.41874547617604285,0.058411377836207895,1.6477776176387566,-1.8233892058032701,11.24196739082057];
        let c01=[-0.0005675007773078625,0.4110745124734173,3.0542232784208947,-0.8600230021038818,-0.4249238161640572,0.057125481763872804,1.649874511858946,-1.8307525435770795,11.240463623349143];
        let c10=[0.4076949454014026,-0.9349780757540036,2.848955652010038,-0.17084970923598725,-0.4539594693472567,-2.492988039824076,4.133178485468616,-0.8082653762310894,10.463988673205847];
        let c11=[0.40164235725767183,-0.9297401988197198,2.85209298544504,-0.17450428704703994,-0.46187029999972007,-2.491937714005673,4.133524222430844,-0.8037283596110483,10.464178121591846];
        let a=polar_segment_parts(c00,c01,[0.5831811428,0.1953667849,-0.0900476649,0.7833409905],[0.5806362033,0.1957665980,-0.0902903527,0.7851017714],4.8e-11).unwrap();
        let b=polar_segment_parts(c10,c11,[-0.4836845994,0.2839072347,-0.1442194730,0.8152586818],[-0.4806513786,0.2843604088,-0.1440301538,0.8169264197],4.8e-11).unwrap();
        // Independent double-precision SVD comparison of the same device Q/C
        // inputs measured 0.1445738 and 0.1976137 rad respectively.
        assert!(a.start>=0.1445738 && b.start>=0.1976137);
        assert!(a.start<0.3792471 && b.start<0.5572502);
        assert!(certified_segment(0.9999,a.start,b.start));
        assert!(certified_segment(0.9999,a.path,b.path));
        assert!(certified_segment(0.9999,a.end,b.end));
    }
    #[test]
    fn covariance_path_bound_does_not_depend_on_inaccurate_device_fit() {
        let c=[12.,0.,0.,0.,1.,0.,0.,0.,1.];
        let angle=1.0_f64;
        let q=[(angle/2.).cos(),(angle/2.).sin(),0.,0.];
        // For this same C and Q the old Q-based symmetric residual exceeds its
        // lower singular bound, although C itself is perfectly nonsingular.
        let h_min=1.0/((1.0_f64/12.0).powi(2)+2.0/angle.cos().powi(2)).sqrt();
        let residual=2.0_f64.sqrt()*angle.sin();
        assert!(residual>h_min);
        let p=polar_segment_parts(c,c,q,q,1e-12).expect("valid polar despite inaccurate fit");
        assert!(p.path<1e-5 && p.start>=angle && p.end>=angle);
        assert!(certified_segment(0.999, p.start,0.0));
    }
    #[test]
    fn accurate_fit_matches_captured_independent_svd_and_rejects_bad_shapes() {
        let c=[0.4076949454_f32,-0.9349780758,2.8489556520,-0.1708497092,-0.4539594693,-2.4929880398,4.1331784855,-0.8082653762,10.4639886732];
        let expected=[-0.5173242874_f64,-0.7640566818,0.3854775852,0.5938641858,-0.6448393357,-0.4811523252,0.6161987589,-0.0199904515,0.7873369490];
        let result=accurate_fit_rotation(c).unwrap();
        for (a,b) in result.iter().zip(expected) {assert!((f64::from(*a)-b).abs()<1e-6);}
        assert!(accurate_fit_rotation([-1.,0.,0.,0.,1.,0.,0.,0.,1.]).is_none());
        assert!(accurate_fit_rotation([1.,0.,0.,0.,1.,0.,0.,0.,0.]).is_none());
        assert!(accurate_fit_rotation([f32::NAN,0.,0.,0.,1.,0.,0.,0.,1.]).is_none());
    }
    #[test]
    fn accurate_fit_handles_real_thin_body_and_certifies_condition_boundary() {
        use nalgebra::{Matrix3,Rotation3,Vector3};
        let captured=[1043333394_u32,3052254256,917788503,866583035,1043750917,1030101545,3023051697,1033403798,1022617879].map(f32::from_bits);
        assert!(accurate_fit_rotation(captured).is_some());
        for scale in [1e-10_f64,1.0,1e10] {for ratio in [0.1,2e-7] {for angle in [0.0,0.37,1.1,2.9] {
            let u=Rotation3::from_euler_angles(angle,angle*0.7,angle*1.3).into_inner();
            let v=Rotation3::from_euler_angles(angle*0.4,-angle*1.1,angle*0.2).into_inner();
            let c=u*Matrix3::from_diagonal(&Vector3::new(scale,scale*0.8,scale*ratio))*v.transpose();
            let raw:[f32;9]=std::array::from_fn(|i|c.as_slice()[i] as f32);
            let fit=accurate_fit_rotation_details(raw).expect("valid rotated covariance");
            let q=Matrix3::from_column_slice(&fit.map(f64::from));
            assert!(q.determinant()>0.0 && (q.transpose()*q-Matrix3::identity()).norm()<=2e-6);
        }}}
        assert!(accurate_fit_rotation([1.,0.,0.,0.,0.8,0.,0.,0.,1e-8]).is_none());
        assert!(accurate_fit_rotation([-1.,0.,0.,0.,1.,0.,0.,0.,1.]).is_none());
        assert!(accurate_fit_rotation([1.,0.,0.,0.,1.,0.,0.,0.,0.]).is_none());
        assert!(accurate_fit_rotation([f32::NAN,0.,0.,0.,1.,0.,0.,0.,1.]).is_none());
    }
    #[test]
    fn relative_bound_certifies_captured_original39_covariances() {
        use nalgebra::Matrix3;
        // Actual private original-replay3 sample 0; expected endpoint angles
        // were independently computed from proper-polar SVD, not this bound.
        let cases=[
            ([0.051452957804638544,-0.008398951774942987,0.01443269360745636,-0.042624918196880746,0.14391688637679323,0.02112062392821712,0.009214716231322417,0.03566146853003982,0.029579919889074935],
             [0.05138682043427978,-0.008532707303102072,0.01459063942566563,-0.041621333020186,0.14455181172577447,0.018819771091530546,0.009451628960833625,0.03606525139085748,0.029019348930594213],2.501762557835046e-13,0.01734442293,0.4920980831),
            ([0.0404654741995264,-0.0004658559605301704,0.005048629658336301,-0.0536435962332511,0.06350226937056114,0.07659091550044383,0.003088488987399607,0.01013174810031403,0.03477580771774935],
             [0.04049173443291887,0.000041622588699441465,0.0048849899476312575,-0.054532783912752736,0.05902184947566891,0.07968290081168194,0.0030259622936655295,0.008598167674983337,0.035228162625755566],1.915031746281988e-13,0.04931337403,2.5849818041),
        ];
        for (a,b,error,actual,old) in cases {
            let c0=Matrix3::from_column_slice(&a);let c1=Matrix3::from_column_slice(&b);
            let bound=relative_polar_path_bound(&c0,&c1,error).unwrap();
            assert!(bound>actual && bound<old*0.1,"{bound} not between {actual} and old {old}");
            for scale in [1e-6,1e6] {
                let scaled=relative_polar_path_bound(&(c0*scale),&(c1*scale),error*scale).unwrap();
                assert!((scaled-bound).abs()<1e-6,"scale {scale} gave {scaled} vs {bound}");
            }
        }
    }
    #[test]
    fn relative_bound_handles_anisotropy_and_rejects_unreliable_inputs() {
        use nalgebra::{Matrix3,Rotation3,Vector3};
        let c0=Matrix3::from_diagonal(&Vector3::new(100.0,1.0,1.0));
        let angle=0.02_f64;let r=Rotation3::from_axis_angle(&Vector3::z_axis(),angle);
        let c1=r.matrix()*c0;
        let bound=relative_polar_path_bound(&c0,&c1,1e-12).unwrap();
        assert!(bound>=angle && bound<0.1);
        let q0=[1.,0.,0.,0.];let q1=[(angle/2.).cos(),0.,0.,(angle/2.).sin()];
        let parts=polar_segment_parts(std::array::from_fn(|i|c0.as_slice()[i]),std::array::from_fn(|i|c1.as_slice()[i]),q0,q1,1e-12).unwrap();
        assert!(parts.path>=angle && parts.path<0.1,"old absolute margin is invalid here");
        assert!(relative_polar_path_bound(&c0,&c0,1e-12).unwrap()<1e-5);
        assert!(relative_polar_path_bound(&Matrix3::zeros(),&c1,0.0).is_none());
        assert!(relative_polar_path_bound(&c0,&c1,10.0).is_none());
        assert!(relative_polar_path_bound(&c0,&(c0*4.0),0.0).is_none());
        let nearly_singular=Matrix3::from_diagonal(&Vector3::new(1.0,1.0,1e-16));
        assert!(relative_polar_path_bound(&nearly_singular,&nearly_singular,1e-8).is_none());
        assert!(relative_polar_path_bound(&Matrix3::repeat(f64::MAX),&c1,0.0).is_none());
        let mut nonfinite=c0;nonfinite[(0,0)]=f64::INFINITY;
        assert!(relative_polar_path_bound(&nonfinite,&c1,0.0).is_none());
        let mut subnormal=c0;subnormal[(0,1)]=f64::from_bits(1);
        assert!(relative_polar_path_bound(&subnormal,&c1,0.0).is_none());
    }
}
