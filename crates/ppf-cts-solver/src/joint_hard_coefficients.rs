//! Hard-only frozen float coefficients, evaluated in host FP64. No assembly,
//! chart, material model, basis probing or solver/preconditioner lives here.
//! The float evaluation certificate follows P -> M -> P^T -> joint atomics.
//! Every nonnegative error computation is rounded upward. MIN_NORMAL covers
//! any FTZ/DAZ implementation; an interval reaching float overflow is refused.
#[repr(C)]
#[derive(Clone,Copy,Default,Debug)]
pub struct SmoothFactors {
    pub body_a:u32,pub body_b:u32,pub ball_active:u32,pub angular_active:u32,
    pub control_active:u32,pub fault:u32,
    pub p_a:[f32;3],pub p_b:[f32;3],pub ball_k:f32,
    pub angular_h:[f32;9],pub control_j:[f32;18],pub control_h:[f32;3],
}
const _:()=assert!(std::mem::size_of::<SmoothFactors>()==172);
const _:()=assert!(std::mem::offset_of!(SmoothFactors,angular_h)==52);
const _:()=assert!(std::mem::offset_of!(SmoothFactors,control_j)==88);

type FatalResult<T>=Result<T,String>;
fn fatal(s:impl std::fmt::Display)->String {format!("G2_HARD_COEFFICIENT_SNAPSHOT: {s}")}

#[derive(Clone,Debug,Default)]
pub struct Csr {
    pub offset:Vec<u32>,pub index:Vec<u32>,pub value:Vec<f32>,
    pub transpose_offset:Vec<u32>,pub transpose_index:Vec<u32>,pub transpose_slot:Vec<u32>,
}
#[derive(Clone,Debug)]
pub struct FrozenCoefficients {
    pub rows:usize,pub dim:usize,pub body_base:usize,pub bodies:usize,
    pub fixed:Csr,pub dynamic:Option<Csr>,pub diagonal:Vec<f32>,
    pub vertex_body:Vec<u32>,pub cloth_offset:Vec<u32>,pub rotated_rest:Vec<f32>,
    pub smooth:Vec<SmoothFactors>,
}

#[derive(Debug,Clone,Copy,Default)]
pub struct LiteralSkew {pub maximum_absolute:f64,pub coefficient_scale:f64,pub block:usize,pub row:usize,pub column:usize}
impl LiteralSkew {
    fn scan(&mut self,blocks:&[f32],row_major:bool,indices:impl Iterator<Item=usize>) {
        for id in indices {let m=&blocks[9*id..9*id+9];
            self.coefficient_scale=self.coefficient_scale.max(m.iter().map(|x|(*x as f64).abs()).fold(0.0,f64::max));
            for r in 0..3 {for c in 0..r {let (i,j)=if row_major {(3*r+c,3*c+r)}else{(3*c+r,3*r+c)};
                let difference=(m[i] as f64-m[j] as f64).abs();
                if difference>self.maximum_absolute {*self=Self{maximum_absolute:difference,coefficient_scale:self.coefficient_scale,block:id,row:r,column:c};}
            }}
        }
    }
}

impl FrozenCoefficients {
    /// Literal float diagonal/H skew, never a basis matrix and never averaged.
    pub fn literal_skew(&self)->[LiteralSkew;4] {
        let mut out=[LiteralSkew::default();4];
        out[0].scan(&self.diagonal,false,0..self.rows);
        let diagonal_ids=|csr:&Csr|->Vec<usize>{(0..self.rows).flat_map(|r|
            (csr.offset[r] as usize..csr.offset[r+1] as usize).filter(move|&s|csr.index[s] as usize==r)).collect()};
        out[1].scan(&self.fixed.value,false,diagonal_ids(&self.fixed).into_iter());
        if let Some(d)=&self.dynamic {out[2].scan(&d.value,false,diagonal_ids(d).into_iter());}
        let h:Vec<_>=self.smooth.iter().flat_map(|f|f.angular_h).collect();
        out[3].scan(&h,true,0..self.smooth.len());out
    }
    pub fn validate(&self)->FatalResult<()> {
        let n=self.rows;
        if self.dim!=self.body_base+6*self.bodies || self.body_base%3!=0 ||
            self.vertex_body.len()!=n || self.cloth_offset.len()!=n ||
            self.rotated_rest.len()!=3*n || self.diagonal.len()!=9*n {
            return Err(fatal("P/M/dimension extent mismatch"));
        }
        for (v,&b) in self.vertex_body.iter().enumerate(){
            if b==u32::MAX {let o=self.cloth_offset[v] as usize;if o%3!=0||o+3>self.body_base{return Err(fatal("cloth slot"));}}
            else if b as usize>=self.bodies{return Err(fatal("body slot"));}
        }
        let mut cloth=vec![false;self.body_base/3];
        for (v,&b) in self.vertex_body.iter().enumerate(){if b==u32::MAX {
            let i=self.cloth_offset[v] as usize/3;
            if std::mem::replace(&mut cloth[i],true){return Err(fatal("duplicate cloth slot"));}
        }}
        if cloth.iter().any(|&x|!x){return Err(fatal("missing cloth slot"));}
        self.fixed.validate(n)?;if let Some(d)=&self.dynamic {d.validate(n)?;}
        if self.diagonal.iter().chain(&self.rotated_rest).any(|x|!x.is_finite()){return Err(fatal("nonfinite float matrix/P"));}
        for f in &self.smooth {
            if f.fault!=0||f.body_a as usize>=self.bodies||f.body_b as usize>=self.bodies||
                f.body_a==f.body_b||f.ball_active>1||f.angular_active>1||f.control_active>1||
                f.p_a.iter().chain(&f.p_b).chain(std::iter::once(&f.ball_k)).chain(&f.angular_h)
                .chain(&f.control_j).chain(&f.control_h).any(|x|!x.is_finite()) {
                return Err(fatal("smooth factor geometry/ABI fault"));
            }
        }
        Ok(())
    }

    /// FP64 evaluation retains the float FACTORS: ball L(p)^T k L(p), isotropic
    /// literal form_h, directional J^T diag(h) J. No rounded dense-H substitution.
    pub fn evaluate(&self,x:&[f64])->FatalResult<Evaluation> {
        if x.len()!=self.dim||x.iter().any(|x|!x.is_finite()){return Err(fatal("apply vector"));}
        let input:Vec<Bound>=x.iter().map(|&x|Bound::input(x)).collect();
        let mut full=vec![Bound::zero();3*self.rows];
        for v in 0..self.rows {
            let b=self.vertex_body[v];
            if b==u32::MAX {let o=self.cloth_offset[v] as usize;full[3*v..3*v+3].copy_from_slice(&input[o..o+3]);}
            else {let o=self.body_base+6*b as usize;
                let t=[input[o],input[o+1],input[o+2]];let r=[input[o+3],input[o+4],input[o+5]];
                let p=&self.rotated_rest[3*v..3*v+3];
                let cross=cross(p,&r);
                for k in 0..3 {full[3*v+k]=t[k].sub(cross[k]);}
            }
        }
        let mut multiplied=vec![Bound::zero();3*self.rows];
        for row in 0..self.rows {for k in 0..3 {
            let mut terms=Vec::new();
            if let Some(d)=&self.dynamic {d.terms(row,k,&full,&mut terms);}
            self.fixed.terms(row,k,&full,&mut terms);
            for c in 0..3 {terms.push(full[3*row+c].scale(self.diagonal[9*row+3*c+k]));}
            // 8-lane strided walk/fold and serial or symmetric scatter each
            // contain at most 2*terms+64 additional f32 additions. This is a
            // topology-dependent bound, not a residual tolerance adjustment.
            multiplied[3*row+k]=sum_any(&terms,2*terms.len()+64);
        }}
        let mut reduced:Vec<Vec<Bound>>=vec![Vec::new();self.dim];
        for v in 0..self.rows {
            let y=[multiplied[3*v],multiplied[3*v+1],multiplied[3*v+2]];
            let b=self.vertex_body[v];
            if b==u32::MAX {let o=self.cloth_offset[v] as usize;for k in 0..3{reduced[o+k].push(y[k]);}}
            else {let o=self.body_base+6*b as usize;let p=&self.rotated_rest[3*v..3*v+3];let torque=cross(p,&y);
                for k in 0..3 {reduced[o+k].push(y[k]);reduced[o+3+k].push(torque[k]);}
            }
        }
        // Restriction atomics finish before joints launch. Collapse P^T once,
        // then bound each family's arbitrary per-owner atomic addition order.
        let mut out:Vec<Bound>=reduced.iter().map(|r|sum_any(r,r.len())).collect();
        for family in 0..3 {
            let mut additions:Vec<Vec<Bound>>=vec![Vec::new();self.dim];
            for f in &self.smooth {
                let a=self.body_base+6*f.body_a as usize;let b=self.body_base+6*f.body_b as usize;
                match family {
                    0 if f.ball_active!=0 => {
                        let la=ball_l(&f.p_a,&input[a..a+6]);let lb=ball_l(&f.p_b,&input[b..b+6]);
                        let w=[la[0].sub(lb[0]),la[1].sub(lb[1]),la[2].sub(lb[2])];
                        let ta=cross(&f.p_a,&w);let tb=cross(&f.p_b,&w);
                        for k in 0..3 {additions[a+k].push(w[k].scale(f.ball_k));
                            additions[b+k].push(w[k].scale(-f.ball_k));
                            additions[a+3+k].push(ta[k].scale(f.ball_k));additions[b+3+k].push(tb[k].scale(-f.ball_k));}
                    },
                    1 if f.angular_active!=0 => {
                        let d=[input[b+3].sub(input[a+3]),input[b+4].sub(input[a+4]),input[b+5].sub(input[a+5])];
                        for k in 0..3 {let t:Vec<_>=(0..3).map(|c|d[c].scale(f.angular_h[3*k+c])).collect();
                            let y=sum_any(&t,t.len());additions[a+3+k].push(y.neg());additions[b+3+k].push(y);}
                    },
                    2 if f.control_active!=0 => {
                        let r=[input[a+3],input[a+4],input[a+5],input[b+3],input[b+4],input[b+5]];
                        let mut z=[Bound::zero();3];
                        for q in 0..3 {let terms:Vec<_>=(0..6).map(|c|r[c].scale(f.control_j[6*q+c])).collect();
                            z[q]=sum_any(&terms,terms.len()).scale(f.control_h[q]);}
                        for k in 0..6 {let terms:Vec<_>=(0..3).map(|q|z[q].scale(f.control_j[6*q+k])).collect();
                            additions[if k<3 {a+3+k}else{b+k}].push(sum_any(&terms,terms.len()));}
                    },_=>{}
                }
            }
            for k in 0..self.dim {if !additions[k].is_empty(){
                additions[k].insert(0,out[k]);out[k]=sum_any(&additions[k],additions[k].len());
            }}
        }
        if out.iter().any(|b|!b.valid()){return Err(fatal("nonfinite or potential f32 overflow in evaluation certificate"));}
        Ok(Evaluation {value:out.iter().map(|x|x.value).collect(),
            float_error:out.iter().map(|x|x.error32).collect(),double_error:out.iter().map(|x|x.error64).collect()})
    }
}

impl Csr {
    fn validate(&self,rows:usize)->FatalResult<()> {
        let offsets=|o:&[u32],len:usize|o.len()==rows+1&&o.first()==Some(&0)&&o.windows(2).all(|w|w[0]<=w[1])&&o.last().copied()==Some(len as u32);
        if !offsets(&self.offset,self.index.len())||self.value.len()!=9*self.index.len()||
            !offsets(&self.transpose_offset,self.transpose_index.len())||self.transpose_slot.len()!=self.transpose_index.len()||
            self.index.iter().chain(&self.transpose_index).any(|&i|i as usize>=rows)||
            self.transpose_slot.iter().any(|&i|i as usize>=self.index.len())||self.value.iter().any(|x|!x.is_finite()) {
            return Err(fatal("malformed CSR direct/transpose/value extent"));
        }
        // References must be EXACT transposes of the stored upper graph. Each
        // off-diagonal appears exactly once in the transpose walk; diagonals
        // never appear there. Rejecting corruption prevents double counting.
        let mut owner=vec![0usize;self.index.len()];let mut seen=vec![0u8;self.index.len()];
        for r in 0..rows {for s in self.offset[r] as usize..self.offset[r+1] as usize {
            owner[s]=r;if (self.index[s] as usize)<r{return Err(fatal("CSR is not upper triangular"));}
        }}
        for r in 0..rows {for e in self.transpose_offset[r] as usize..self.transpose_offset[r+1] as usize {
            let s=self.transpose_slot[e] as usize;
            if self.transpose_index[e] as usize!=owner[s]||self.index[s] as usize!=r||owner[s]>=r||seen[s]!=0{return Err(fatal("inconsistent transpose reference"));}
            seen[s]=1;
        }}
        if seen.iter().enumerate().any(|(s,&v)|v!=u8::from(self.index[s] as usize!=owner[s])) {return Err(fatal("missing transpose reference"));}
        Ok(())
    }
    fn terms(&self,row:usize,k:usize,x:&[Bound],out:&mut Vec<Bound>){
        for s in self.offset[row] as usize..self.offset[row+1] as usize {for c in 0..3 {
            out.push(x[3*self.index[s] as usize+c].scale(self.value[9*s+3*c+k]));
        }}
        for e in self.transpose_offset[row] as usize..self.transpose_offset[row+1] as usize {let s=self.transpose_slot[e] as usize;for c in 0..3 {
            out.push(x[3*self.transpose_index[e] as usize+c].scale(self.value[9*s+3*k+c]));
        }}
    }
}

#[derive(Debug)]
pub struct Evaluation {pub value:Vec<f64>,pub float_error:Vec<f64>,pub double_error:Vec<f64>}
#[derive(Debug)]
pub struct Certificate {pub raw_relative_l1:f64,pub mathematical_relative_l1:f64,pub max_bound_ratio:f64,pub raw_passed:bool}
impl Evaluation {
    pub fn mathematical_relative_l1(&self,rhs:&[f64])->FatalResult<f64> {
        if rhs.len()!=self.value.len()||rhs.iter().any(|x|!x.is_finite()){return Err(fatal("mathematical residual extent/nonfinite"));}
        let norm=rhs.iter().fold(0.0,|s,x|down(s+x.abs()));
        let residual=self.value.iter().zip(rhs).zip(&self.double_error).fold(0.0,|s,((x,b),e)|
            plus(s,plus(plus((x-b).abs(),times(U64,plus(x.abs(),b.abs()))),*e)));
        Ok(if norm==0.0 {if residual==0.0 {0.0}else{f64::INFINITY}}else{up(residual/norm)})
    }
    pub fn certify_roundoff(&self,gpu:&[f64])->FatalResult<f64> {
        if gpu.len()!=self.value.len()||gpu.iter().any(|x|!x.is_finite()){return Err(fatal("roundoff certificate extent/nonfinite"));}
        let mut ratio:f64=0.0;
        for k in 0..gpu.len(){let difference=up((gpu[k]-self.value[k]).abs());let bound=up(self.float_error[k]+self.double_error[k]);
            if difference>bound{return Err(fatal(format!("GPU component {k} difference {difference:.12e} exceeds roundoff bound {bound:.12e}")));}
            if bound>0.0{ratio=ratio.max(difference/bound);}
        }
        Ok(ratio)
    }
    /// Certify the actual float transported vector. A failed RAW gate stays
    /// failed in the receipt; only coefficient residual + component certificate
    /// together permit the explicit FAILED_RAW_CERTIFIED_ROUNDOFF contract.
    pub fn certify(&self,gpu:&[f64],rhs:&[f64],tol:f64)->FatalResult<Certificate> {
        if gpu.len()!=self.value.len()||rhs.len()!=gpu.len()||gpu.iter().chain(rhs).any(|x|!x.is_finite()) {return Err(fatal("certificate extent/nonfinite"));}
        let norm=rhs.iter().fold(0.0,|s,x|down(s+x.abs()));
        let raw=gpu.iter().zip(rhs).map(|(x,b)|(x-b).abs()).sum::<f64>();
        let raw_relative=if norm==0.0 {if raw==0.0 {0.0}else{f64::INFINITY}}else{raw/norm};
        let math_relative=self.mathematical_relative_l1(rhs)?;
        if !math_relative.is_finite()||math_relative>tol{return Err(fatal(format!("coefficient64 transported residual {math_relative:.12e} > {tol:.12e}")));}
        let ratio=self.certify_roundoff(gpu)?;
        Ok(Certificate {raw_relative_l1:raw_relative,mathematical_relative_l1:math_relative,
            max_bound_ratio:ratio,raw_passed:raw_relative<=tol})
    }
}

pub struct FloatPublication {
    pub values:Vec<f64>,pub nearest_relative_l1:f64,pub selected_relative_l1:f64,
    pub evaluations:usize,pub changed_lanes:usize,pub absolute_transport_budget:f64,
    pub lane_budgets:Vec<f64>,pub lane_group_norms:Vec<f64>,
}

/// A representable transport under the natural vector cast infinity budget:
/// |q_i-x_i| <= up(u32*max_{same units}|x| + MIN_NORMAL32), for EVERY lane.
/// Flexible/rigid translations use meters; rigid rotations use radians.
/// already a bound on nearest float vector casting including FTZ, not a new
/// physical/force tolerance. Small lanes may compensate coupled quantization
/// error within this same vector budget. Every candidate is a FLOAT vector,
/// evaluated by the original frozen coefficient apply plus its FP64 bound.
/// At most 4096 full applies / three sweeps. No GPU basis matrix, dense solve,
/// coefficient averaging, force/pose change or material reevaluation occurs.
/// Returned vectors must satisfy the original mathematical residual gate;
/// actual GPU certificate and nonlinear candidate geometry remain caller gates.
pub fn select_float_publication(snapshot:&FrozenCoefficients,x:&[f64],rhs:&[f64],limit:f64)
    ->FatalResult<FloatPublication> {
    if x.len()!=snapshot.dim||rhs.len()!=x.len()||!limit.is_finite()||limit<=0.0{return Err(fatal("float publication extent/limit"));}
    let nearest:Vec<f32>=x.iter().map(|&v|v as f32).collect();
    if x.iter().any(|v|!v.is_finite())||nearest.iter().any(|v|!v.is_finite()){return Err(fatal("float publication overflow/nonfinite"));}
    let is_translation=|lane:usize|lane<snapshot.body_base||(lane-snapshot.body_base)%6<3;
    let mut norms=[0.0_f64;2];for (lane,v) in x.iter().enumerate(){let group=usize::from(!is_translation(lane));norms[group]=norms[group].max(v.abs());}
    let lane_norms:Vec<_>=(0..x.len()).map(|lane|norms[usize::from(!is_translation(lane))]).collect();
    let budgets:Vec<_>=lane_norms.iter().map(|&s|plus(times(U32,s),FTZ)).collect();
    let budget=budgets.iter().copied().fold(0.0,f64::max);
    let mut values:Vec<f64>=nearest.iter().map(|&v|v as f64).collect();
    if values.iter().zip(x).zip(&budgets).any(|((q,x),b)|(q-x).abs()>*b){return Err(fatal("nearest float exceeds same-unit natural transport budget"));}
    let mut best=snapshot.evaluate(&values)?.mathematical_relative_l1(rhs)?;
    let before=best;let mut evaluations=1;
    if best>limit {'sweeps:for _ in 0..3 {
        let start=best;
        for lane in 0..values.len() {
            // Descending dyadic steps find a representable residual correction
            // without differentiating L1 or substituting a linearized score.
            for divisor in [1.,2.,4.,8.,16.,32.,64.,128.,256.,512.] {
                for sign in [-1.,1.] {
                    let candidate=((values[lane]+sign*budgets[lane]/divisor) as f32) as f64;
                    if !candidate.is_finite()||candidate==values[lane]||up((candidate-x[lane]).abs())>budgets[lane] {continue;}
                    if evaluations==4096{break 'sweeps;}
                    let old=values[lane];values[lane]=candidate;
                    let score=snapshot.evaluate(&values)?.mathematical_relative_l1(rhs)?;evaluations+=1;
                    if score<best {best=score;}else{values[lane]=old;}
                    if best<=limit {break 'sweeps;}
                }
            }
        }
        if best>=start {break;}
    }}
    if !best.is_finite()||best>limit {
        // Failure-only evidence, not a relaxation or another search. Values
        // are finite by validation above; Rust's shortest float spelling is
        // lossless JSON. Context/stamp and solved rows are emitted by caller.
        if std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some() {
            let nearest_wide:Vec<f64>=nearest.iter().map(|&q|q as f64).collect();
            let high=snapshot.evaluate(x)?;
            let near=snapshot.evaluate(&nearest_wide)?;
            let selected=snapshot.evaluate(&values)?;
            let nearest_delta:Vec<_>=nearest_wide.iter().zip(x).map(|(q,x)|q-x).collect();
            let best_delta:Vec<_>=values.iter().zip(x).map(|(q,x)|q-x).collect();
            eprintln!("HARD_FLOAT_PUBLICATION_FAILURE_JSON {{\"kind\":\"hard_float_publication_failure\",\"dimension\":{},\"body_base\":{},\"high64\":{:?},\"rhs\":{:?},\"nearest_float\":{:?},\"best_float\":{:?},\"nearest_delta\":{:?},\"best_delta\":{:?},\"lane_budgets\":{:?},\"lane_same_unit_infinity_norms\":{:?},\"high64_apply\":{:?},\"high64_eval_bound\":{:?},\"nearest_apply\":{:?},\"nearest_eval_bound\":{:?},\"best_apply\":{:?},\"best_eval_bound\":{:?},\"high64_math_relative_l1\":{},\"nearest_math_relative_l1\":{},\"best_math_relative_l1\":{},\"limit\":{},\"evaluations\":{},\"scope\":\"diagnostic bounded-search failure; no vector published; no proof of lattice infeasibility\"}}",snapshot.dim,snapshot.body_base,x,rhs,nearest_wide,values,nearest_delta,best_delta,budgets,lane_norms,high.value,high.double_error,near.value,near.double_error,selected.value,selected.double_error,high.mathematical_relative_l1(rhs)?,before,best,limit,evaluations);
        }
        return Err(fatal(format!("representable publication exhausted evaluations={evaluations} nearest={before:.12e} best={best:.12e} limit={limit:.12e} absolute_transport_budget={budget:.12e}")));
    }
    let changed=values.iter().zip(&nearest).filter(|(a,b)|**a!=**b as f64).count();
    Ok(FloatPublication{values,nearest_relative_l1:before,selected_relative_l1:best,evaluations,changed_lanes:changed,absolute_transport_budget:budget,lane_budgets:budgets,lane_group_norms:lane_norms})
}

// Upward operations on nonnegative finite bounds. A nonfinite bound propagates
// and makes valid() fail; no overflow can become a permissive infinite bound.
fn up(x:f64)->f64 {if x==0.0 {0.0}else if x.is_finite()&&x>=0.0 {f64::from_bits(x.to_bits()+1)}else{x}}
fn down(x:f64)->f64 {if x>0.0&&x.is_finite(){f64::from_bits(x.to_bits()-1)}else{x}}
fn plus(a:f64,b:f64)->f64 {up(a+b)}
fn times(a:f64,b:f64)->f64 {up(a*b)}
fn gamma(n:usize,u:f64)->f64 {let nu=times(n as f64,u);if nu>=1.0 {f64::INFINITY}else{up(nu/down(1.0-nu))}}
const U32:f64=1.0/16777216.0;
const U64:f64=1.0/9007199254740992.0;
const FTZ:f64=f32::MIN_POSITIVE as f64;

#[derive(Clone,Copy,Debug)]
struct Bound {value:f64,error32:f64,error64:f64,magnitude32:f64,overflow:bool}
impl Bound {
    fn zero()->Self {Self{value:0.0,error32:0.0,error64:0.0,magnitude32:0.0,overflow:false}}
    fn input(value:f64)->Self {
        let transported=value as f32 as f64;
        let cast=up((transported-value).abs());
        let daz=if transported!=0.0&&transported.abs()<FTZ {transported.abs()}else{0.0};
        Self{value,error32:plus(cast,daz),error64:0.0,magnitude32:plus(value.abs(),plus(cast,daz)),overflow:!transported.is_finite()}
    }
    fn neg(mut self)->Self {self.value=-self.value;self}
    fn scale(self,a:f32)->Self {
        let a=a as f64;let value=self.value*a;
        let m32=times(a.abs(),plus(self.value.abs(),self.error32));
        let local32=plus(plus(times(U32,m32),if self.value!=0.0&&a!=0.0 {FTZ}else{0.0}),if a!=0.0&&a.abs()<FTZ {m32}else{0.0});
        let local64=plus(times(U64,times(self.value.abs(),a.abs())),if self.value!=0.0&&a!=0.0 {f64::MIN_POSITIVE}else{0.0});
        Self{value,error32:plus(plus(times(a.abs(),self.error32),local32),local64),
            error64:plus(times(a.abs(),self.error64),local64),magnitude32:plus(m32,local32),
            overflow:self.overflow||plus(m32,local32)>=f32::MAX as f64}
    }
    fn sub(self,b:Self)->Self {sum_any(&[self,b.neg()],2)}
    fn valid(&self)->bool {!self.overflow&&self.value.is_finite()&&self.error32.is_finite()&&self.error64.is_finite()&&
        self.magnitude32.is_finite()&&plus(self.value.abs(),self.error32)<f32::MAX as f64}
}
fn sum_any(terms:&[Bound],float_additions:usize)->Bound {
    if terms.is_empty(){return Bound::zero();}
    let mut value=0.0;let mut err32=0.0;let mut err64=0.0;let mut abs32=0.0;let mut abs64=0.0;
    for t in terms {value+=t.value;err32=plus(err32,t.error32);err64=plus(err64,t.error64);
        abs32=plus(abs32,plus(t.value.abs(),t.error32));abs64=plus(abs64,t.value.abs());}
    let round64=times(gamma(terms.len(),U64),abs64);
    let round32=plus(times(gamma(float_additions,U32),abs32),times(float_additions as f64,FTZ));
    Bound{value,error32:plus(plus(err32,round32),round64),error64:plus(err64,round64),magnitude32:plus(abs32,round32),
        overflow:terms.iter().any(|t|t.overflow)||plus(abs32,round32)>=f32::MAX as f64}
}
fn cross(p:&[f32],v:&[Bound;3])->[Bound;3] {
    [v[2].scale(p[1]).sub(v[1].scale(p[2])),v[0].scale(p[2]).sub(v[2].scale(p[0])),v[1].scale(p[0]).sub(v[0].scale(p[1]))]
}
fn ball_l(p:&[f32;3],u:&[Bound])->[Bound;3] {
    // Literal ball::L spelling (its cross sign differs from PDRD).
    [sum_any(&[u[0],u[5].scale(-p[1]),u[4].scale(p[2])],2),
     sum_any(&[u[1],u[5].scale(p[0]),u[3].scale(-p[2])],2),
     sum_any(&[u[2],u[4].scale(-p[0]),u[3].scale(p[1])],2)]
}

/// Mixed precision iterative refinement. Approximate correction uses the
/// existing float solver; the accumulator and the residual NEVER narrow.
/// The tighter internal target preserves the existing 1e-11 Gram symmetry
/// guard. Failure at any bounded iteration is explicit, never a relaxed gate.
pub fn refine<F>(snapshot:&FrozenCoefficients,rhs:&[f64],initial:&[f64],mathematical_limit:f64,context:&str,mut approximate:F)
    ->Result<Vec<f64>,String> where F:FnMut(&[f64])->Result<Vec<f64>,String> {
    if rhs.len()!=snapshot.dim||initial.len()!=snapshot.dim{return Err(fatal("refinement extent"));}
    let norm=rhs.iter().fold(0.0,|s,x|down(s+x.abs()));
    if !norm.is_finite()||!mathematical_limit.is_finite()||mathematical_limit<=0.0{return Err(fatal("refinement RHS norm/limit"));}
    if norm==0.0{return Ok(vec![0.0;rhs.len()]);}
    let mut x=initial.to_vec();let mut last=f64::INFINITY;
    let diagnostic=std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some();
    let first_rhs=rhs.iter().position(|x|*x!=0.0).unwrap_or(rhs.len());
    for round in 0..=6 {
        let e=snapshot.evaluate(&x)?;
        let residual:Vec<_>=rhs.iter().zip(&e.value).map(|(b,a)|b-a).collect();
        let actual=residual.iter().fold(0.0,|s,r|plus(s,r.abs()));
        let mut subtraction=0.0;let mut floor=0.0;let mut unresolved=true;
        for k in 0..rhs.len(){
            let sub=times(U64,plus(rhs[k].abs(),e.value[k].abs()));
            subtraction=plus(subtraction,sub);floor=plus(floor,e.double_error[k]);
            // A component whose interval includes zero is below the evaluator's
            // resolving power. This does not turn its upper residual into zero.
            unresolved &= residual[k].abs()<=plus(e.double_error[k],sub);
        }
        let point=up(plus(actual,subtraction)/norm);
        let floor_relative=up(floor/norm);
        let combined=up(plus(plus(actual,subtraction),floor)/norm);
        if !point.is_finite()||!combined.is_finite(){return Err(fatal("nonfinite FP64 refinement residual"));}
        if diagnostic {
            eprintln!("HARD_FP64_REFINEMENT context={context} round={round} rhs_first_nonzero={first_rhs} rhs_l1_lower={norm:.12e} actual_relative_l1={:.12e} subtraction_bound_relative_l1={:.12e} point_bound_relative_l1={point:.12e} evaluation_floor_relative_l1={floor_relative:.12e} combined_relative_l1={combined:.12e} internal_point_target=1.000000000000e-13 mathematical_limit={mathematical_limit:.12e} all_component_intervals_include_zero={unresolved}",actual/norm,subtraction/norm);
        }
        // The internal goal belongs to the reducible POINT residual. The
        // independent evaluation floor is NOT something inverse refinement can
        // lower. Both stopping reasons still require the unchanged certified
        // outer mathematical limit, followed by the existing strict Gram gate.
        if combined<=mathematical_limit&&(point<=1e-13||unresolved){
            if diagnostic {eprintln!("HARD_FP64_REFINEMENT_STOP context={context} round={round} reason={} point_bound_relative_l1={point:.12e} evaluation_floor_relative_l1={floor_relative:.12e} combined_relative_l1={combined:.12e}",if point<=1e-13 {"POINT_TARGET_CERTIFIED"}else{"EVALUATOR_RESOLUTION_CERTIFIED"});}
            return Ok(x);
        }
        if round==6||point>=last {
            return Err(fatal(format!("FP64 refinement unresolved/stagnated context={context} round={round} point={point:.12e} evaluation_floor={floor_relative:.12e} combined={combined:.12e} mathematical_limit={mathematical_limit:.12e}")));
        }
        last=point;
        let delta=approximate(&residual)?;
        if delta.len()!=x.len()||delta.iter().any(|d|!d.is_finite()){return Err(fatal("inverse correction vector"));}
        for (x,d) in x.iter_mut().zip(delta){*x+=d;}
    }
    unreachable!()
}





pub const PUBLICATION_FORCE_QUALITY_CEILING:f64=1e-6;
pub const PUBLICATION_VERTEX_CAST_CAP_M:f64=1e-8;
pub const PUBLICATION_ROTATION_CAST_CAP_RAD:f64=1e-8;
pub struct NearestPublication {
    pub values:Vec<f64>,pub deltas:Vec<f64>,pub lane_budgets:Vec<f64>,
    pub propagated_component_bounds:Vec<f64>,pub fp64_relative_l1:f64,
    pub float_relative_l1:f64,pub envelope_relative_l1:f64,
    pub vertex_cast_bound_max:f64,pub rotation_cast_bound_max:f64,
    pub cast_action_component_bounds:Vec<f64>,pub delta_eval_value:Vec<f64>,
    pub delta_eval_error:Vec<f64>,pub delta_subtraction_bounds:Vec<f64>,
    pub delta_subtraction_propagated:Vec<f64>,pub worst_budget_envelope_relative_l1:f64,
    pub cast_identity_component_bounds:Vec<f64>,pub cast_identity_mismatches:Vec<f64>,
    pub cast_identity_maximum_bound_ratio:f64,
}
fn positive_cross(p:&[f32],v:&[Bound;3])->[Bound;3]{
    [sum_any(&[v[2].scale(p[1].abs()),v[1].scale(p[2].abs())],2),
     sum_any(&[v[0].scale(p[2].abs()),v[2].scale(p[0].abs())],2),
     sum_any(&[v[1].scale(p[0].abs()),v[0].scale(p[1].abs())],2)]
}
fn positive_ball(p:&[f32;3],u:&[Bound])->[Bound;3]{
    [sum_any(&[u[0],u[5].scale(p[1].abs()),u[4].scale(p[2].abs())],2),
     sum_any(&[u[1],u[5].scale(p[0].abs()),u[3].scale(p[2].abs())],2),
     sum_any(&[u[2],u[4].scale(p[0].abs()),u[3].scale(p[1].abs())],2)]
}
impl FrozenCoefficients {
    /// Reconcile literal G skew with the exact recorded coefficient transpose
    /// and measured inverse residuals. Bounds cover each FP64 evaluate/dot/
    /// subtraction. This does not certify float assembly formation accuracy.
    pub fn audit_literal_gram(&self,b:&[Vec<f64>],y:&[Vec<f64>],g:&[Vec<f64>])->FatalResult<()> {
        let n=b.len();if y.len()!=n||g.len()!=n||b.iter().chain(y).any(|v|v.len()!=self.dim)||g.iter().any(|v|v.len()!=n){return Err(fatal("Gram audit extent"));}
        let transpose=self.transposed_literal();let ky:Vec<_>=y.iter().map(|v|self.evaluate(v)).collect::<FatalResult<_>>()?;let kty:Vec<_>=y.iter().map(|v|transpose.evaluate(v)).collect::<FatalResult<_>>()?;
        let mut max_ratio=0.0_f64;
        for i in 0..n {for j in 0..n {
            let mut terms=Vec::new();let mut error=0.;let mut absolute=0.;
            for k in 0..self.dim {
                let antisym=kty[j].value[k]-ky[j].value[k];let ri=b[i][k]-ky[i].value[k];let rj=b[j][k]-ky[j].value[k];
                let ea=plus(plus(kty[j].double_error[k],ky[j].double_error[k]),times(U64,plus(kty[j].value[k].abs(),ky[j].value[k].abs())));
                let ei=plus(ky[i].double_error[k],times(U64,plus(b[i][k].abs(),ky[i].value[k].abs())));
                let ej=plus(ky[j].double_error[k],times(U64,plus(b[j][k].abs(),ky[j].value[k].abs())));
                terms.extend([y[i][k]*antisym,ri*y[j][k],-rj*y[i][k]]);
                error=plus(error,plus(times(y[i][k].abs(),ea),plus(times(y[j][k].abs(),ei),times(y[i][k].abs(),ej))));
                absolute=plus(absolute,plus(times(b[i][k].abs(),y[j][k].abs()),times(b[j][k].abs(),y[i][k].abs())));
            }
            let predicted=terms.iter().sum::<f64>();let magnitude=terms.iter().fold(0.,|s,t|plus(s,t.abs()));
            let bound=plus(error,plus(times(gamma(2*self.dim+4,U64),absolute),times(gamma(2*terms.len()+4,U64),magnitude)));
            let lhs=g[i][j]-g[j][i];let complete=plus(bound,times(U64,plus(plus(g[i][j].abs(),g[j][i].abs()),predicted.abs())));
            let mismatch=up((lhs-predicted).abs());if !mismatch.is_finite()||!complete.is_finite()||mismatch>complete{return Err(fatal(format!("literal Gram skew identity failed pair={i}/{j} difference={mismatch:.12e} bound={complete:.12e}")));}
            if complete>0.{max_ratio=max_ratio.max(mismatch/complete);}
        }}
        eprintln!("HARD_LITERAL_SKEW_IDENTITY status=PASSED maximum_component_bound_ratio={max_ratio:.12e} scope=literal_coefficient_transpose_and_actual_inverse_residual_not_assembly_formation_proof");Ok(())
    }
    /// Absolute FACTOR propagation, not row curvature and not GPU basis A.
    /// Includes evaluator FP64 upper bounds on |P|^T |M| |P|d and each original
    /// smooth factor family. Conservative cancellation loss is retained.
    pub fn absolute_transport_bound(&self,d:&[f64])->FatalResult<Vec<f64>> {
        if d.len()!=self.dim||d.iter().any(|x|!x.is_finite()||*x<0.){return Err(fatal("absolute transport extent/nonfinite"));}
        let input:Vec<_>=d.iter().map(|&x|Bound::input(x)).collect();let mut full=vec![Bound::zero();3*self.rows];
        for v in 0..self.rows {let b=self.vertex_body[v];if b==u32::MAX{let o=self.cloth_offset[v] as usize;full[3*v..3*v+3].copy_from_slice(&input[o..o+3]);}
            else {let o=self.body_base+6*b as usize;let r=[input[o+3],input[o+4],input[o+5]];let c=positive_cross(&self.rotated_rest[3*v..3*v+3],&r);for k in 0..3{full[3*v+k]=sum_any(&[input[o+k],c[k]],2);}}
        }
        let mut m=vec![Bound::zero();3*self.rows];
        for r in 0..self.rows {for k in 0..3 {let mut terms=Vec::new();
            for csr in std::iter::once(&self.fixed).chain(self.dynamic.iter()) {
                for s in csr.offset[r] as usize..csr.offset[r+1] as usize{for c in 0..3{terms.push(full[3*csr.index[s] as usize+c].scale(csr.value[9*s+3*c+k].abs()));}}
                for e in csr.transpose_offset[r] as usize..csr.transpose_offset[r+1] as usize{let s=csr.transpose_slot[e] as usize;for c in 0..3{terms.push(full[3*csr.transpose_index[e] as usize+c].scale(csr.value[9*s+3*k+c].abs()));}}
            }
            for c in 0..3{terms.push(full[3*r+c].scale(self.diagonal[9*r+3*c+k].abs()));}
            m[3*r+k]=sum_any(&terms,2*terms.len()+64);
        }}
        let mut terms=vec![Vec::<Bound>::new();self.dim];
        for v in 0..self.rows {let b=self.vertex_body[v];let y=[m[3*v],m[3*v+1],m[3*v+2]];
            if b==u32::MAX{let o=self.cloth_offset[v] as usize;for k in 0..3{terms[o+k].push(y[k]);}}
            else {let o=self.body_base+6*b as usize;let c=positive_cross(&self.rotated_rest[3*v..3*v+3],&y);for k in 0..3{terms[o+k].push(y[k]);terms[o+3+k].push(c[k]);}}
        }
        for f in &self.smooth {let a=self.body_base+6*f.body_a as usize;let b=self.body_base+6*f.body_b as usize;
            if f.ball_active!=0{let la=positive_ball(&f.p_a,&input[a..a+6]);let lb=positive_ball(&f.p_b,&input[b..b+6]);let w=[sum_any(&[la[0],lb[0]],2),sum_any(&[la[1],lb[1]],2),sum_any(&[la[2],lb[2]],2)];let ca=positive_cross(&f.p_a,&w);let cb=positive_cross(&f.p_b,&w);
                for k in 0..3{terms[a+k].push(w[k].scale(f.ball_k.abs()));terms[b+k].push(w[k].scale(f.ball_k.abs()));terms[a+3+k].push(ca[k].scale(f.ball_k.abs()));terms[b+3+k].push(cb[k].scale(f.ball_k.abs()));}}
            if f.angular_active!=0{let d=[sum_any(&[input[a+3],input[b+3]],2),sum_any(&[input[a+4],input[b+4]],2),sum_any(&[input[a+5],input[b+5]],2)];for k in 0..3{let t:Vec<_>=(0..3).map(|c|d[c].scale(f.angular_h[3*k+c].abs())).collect();let y=sum_any(&t,t.len());terms[a+3+k].push(y);terms[b+3+k].push(y);}}
            if f.control_active!=0{let r=[input[a+3],input[a+4],input[a+5],input[b+3],input[b+4],input[b+5]];let mut z=[Bound::zero();3];for q in 0..3{let t:Vec<_>=(0..6).map(|c|r[c].scale(f.control_j[6*q+c].abs())).collect();z[q]=sum_any(&t,t.len()).scale(f.control_h[q].abs());}
                for k in 0..6{let t:Vec<_>=(0..3).map(|q|z[q].scale(f.control_j[6*q+k].abs())).collect();terms[if k<3{a+3+k}else{b+k}].push(sum_any(&t,t.len()));}}
        }
        let out:Vec<_>=terms.iter().map(|t|sum_any(t,t.len())).collect();
        if out.iter().any(|t|!t.valid()){return Err(fatal("absolute propagation overflow/nonfinite"));}
        Ok(out.iter().map(|t|plus(t.value,t.error64)).collect())
    }
    /// Transposition here only audits the literal skew identity. Every original
    /// coefficient bit is retained; no material/chart is re-evaluated.
    pub fn transposed_literal(&self)->Self {
        let mut out=self.clone();
        fn transpose(m:&mut[f32]){for r in 0..3{for c in 0..r{m.swap(3*c+r,3*r+c);}}}
        for m in out.diagonal.chunks_exact_mut(9){transpose(m);}
        for csr in std::iter::once(&mut out.fixed).chain(out.dynamic.iter_mut()){for r in 0..out.rows{for s in csr.offset[r] as usize..csr.offset[r+1] as usize{if csr.index[s] as usize==r{transpose(&mut csr.value[9*s..9*s+9]);}}}}
        for f in &mut out.smooth{transpose(&mut f.angular_h);}out
    }
    pub fn certify_nearest_publication(&self,x:&[f64],rhs:&[f64],limit:f64)->FatalResult<NearestPublication>{
        let high=self.evaluate(x)?;let high_relative=high.mathematical_relative_l1(rhs)?;
        if !high_relative.is_finite()||high_relative>limit{return Err(fatal("used FP64 constrained equilibrium failed"));}
        let values:Vec<_>=x.iter().map(|&x|x as f32 as f64).collect();if values.iter().any(|q|!q.is_finite()){return Err(fatal("nearest publication nonfinite"));}
        let deltas:Vec<_>=values.iter().zip(x).map(|(q,x)|q-x).collect();
        let budgets:Vec<_>=x.iter().map(|x|plus(times(U32,x.abs()),FTZ)).collect();
        if values.iter().zip(x).zip(&budgets).any(|((q,x),d)|plus((q-x).abs(),times(U64,plus(q.abs(),x.abs())))>*d){return Err(fatal("nearest actual cast exceeds lane rounding budget"));}
        // Exactly zero input has exactly zero cast and linear operator output;
        // relative zero-RHS residual is zero only in this exact identity case.
        if rhs.iter().all(|v|*v==0.)&&x.iter().all(|v|*v==0.){
            return Ok(NearestPublication{values,deltas,lane_budgets:budgets,propagated_component_bounds:vec![0.;self.dim],fp64_relative_l1:0.,float_relative_l1:0.,envelope_relative_l1:0.,vertex_cast_bound_max:0.,rotation_cast_bound_max:0.,cast_action_component_bounds:vec![0.;self.dim],delta_eval_value:vec![0.;self.dim],delta_eval_error:vec![0.;self.dim],delta_subtraction_bounds:vec![0.;self.dim],delta_subtraction_propagated:vec![0.;self.dim],worst_budget_envelope_relative_l1:0.,cast_identity_component_bounds:vec![0.;self.dim],cast_identity_mismatches:vec![0.;self.dim],cast_identity_maximum_bound_ratio:0.});
        }
        let mut vertex_max=0.0_f64;let mut rotation_max=0.0_f64;
        // Euclidean upper bound via L1 is valid for each physical 3-vector.
        // Propagate PREDECLARED d, stronger than only checking favorable delta.
        for v in 0..self.rows{let b=self.vertex_body[v];let upper=if b==u32::MAX{let o=self.cloth_offset[v] as usize;budgets[o..o+3].iter().fold(0.,|s,x|plus(s,*x))}
            else{let o=self.body_base+6*b as usize;let r=[Bound::input(budgets[o+3]),Bound::input(budgets[o+4]),Bound::input(budgets[o+5])];let c=positive_cross(&self.rotated_rest[3*v..3*v+3],&r);(0..3).fold(0.,|s,k|plus(s,plus(budgets[o+k],plus(c[k].value,c[k].error64))))};vertex_max=vertex_max.max(upper);}
        for b in 0..self.bodies{let o=self.body_base+6*b+3;rotation_max=rotation_max.max(budgets[o..o+3].iter().fold(0.,|s,x|plus(s,*x)));}
        if vertex_max>PUBLICATION_VERTEX_CAST_CAP_M||rotation_max>PUBLICATION_ROTATION_CAST_CAP_RAD{return Err(fatal(format!("nearest cast physical cap vertex={vertex_max:.12e} rotation={rotation_max:.12e}")));}
        // d describes ALL permitted cast perturbations. For this single fixed
        // nearest q we certify its actual signed action, retaining K's real
        // cancellation. delta_hat subtraction is NOT presumed exact.
        let propagated=self.absolute_transport_bound(&budgets)?;let q_eval=self.evaluate(&values)?;
        let delta_eval=self.evaluate(&deltas)?;
        let eta:Vec<_>=values.iter().zip(x).map(|(q,x)|plus(times(U64,plus(q.abs(),x.abs())),f64::MIN_POSITIVE)).collect();
        let eta_action=self.absolute_transport_bound(&eta)?;
        let actual_action:Vec<_>=delta_eval.value.iter().zip(&delta_eval.double_error).zip(&eta_action)
            .map(|((value,error),eta)|plus(plus(value.abs(),*error),*eta)).collect();
        // Independent signed linearity audit. Triangle residual bounds alone
        // cannot catch a wrong delta/factor path hidden by a large allowance.
        let mut identity_bounds=Vec::with_capacity(self.dim);let mut identity_mismatches=Vec::with_capacity(self.dim);let mut identity_ratio=0.0_f64;
        for k in 0..self.dim{
            let difference=q_eval.value[k]-high.value[k];let mismatch=(difference-delta_eval.value[k]).abs();
            let round_first=plus(times(U64,plus(q_eval.value[k].abs(),high.value[k].abs())),f64::MIN_POSITIVE);
            let round_second=plus(times(U64,plus(difference.abs(),delta_eval.value[k].abs())),f64::MIN_POSITIVE);
            let bound=plus(plus(plus(q_eval.double_error[k],high.double_error[k]),plus(delta_eval.double_error[k],eta_action[k])),plus(round_first,round_second));
            if !mismatch.is_finite()||!bound.is_finite()||mismatch>bound{return Err(fatal(format!("nearest signed cast identity component {k} mismatch={mismatch:.12e} bound={bound:.12e}")));}
            identity_ratio=identity_ratio.max(if bound>0.{up(mismatch/bound)}else{0.});identity_bounds.push(bound);identity_mismatches.push(mismatch);
        }
        let norm=rhs.iter().fold(0.,|s,x|down(s+x.abs()));if norm<=0.{return Err(fatal("publication RHS norm"));}
        let mut envelope=0.;let mut worst_envelope=0.;
        for k in 0..self.dim{
            let high_bound=plus(plus((rhs[k]-high.value[k]).abs(),times(U64,plus(rhs[k].abs(),high.value[k].abs()))),high.double_error[k]);
            let q_error=plus(q_eval.double_error[k],times(U64,plus(rhs[k].abs(),q_eval.value[k].abs())));
            let envelope_k=plus(plus(high_bound,actual_action[k]),q_error);
            worst_envelope=plus(worst_envelope,plus(plus(high_bound,propagated[k]),q_error));
            let actual=plus((rhs[k]-q_eval.value[k]).abs(),times(U64,plus(rhs[k].abs(),q_eval.value[k].abs())));
            if actual>envelope_k{return Err(fatal(format!("nearest coefficient component {k} exceeds predeclared cast envelope")));}
            envelope=plus(envelope,envelope_k);
        }
        let relative=up(envelope/norm);if !relative.is_finite()||relative>PUBLICATION_FORCE_QUALITY_CEILING{return Err(fatal(format!("nearest cast envelope quality {relative:.12e} > {PUBLICATION_FORCE_QUALITY_CEILING:.12e}")));}
        Ok(NearestPublication{values,deltas,lane_budgets:budgets,propagated_component_bounds:propagated,fp64_relative_l1:high_relative,float_relative_l1:q_eval.mathematical_relative_l1(rhs)?,envelope_relative_l1:relative,vertex_cast_bound_max:vertex_max,rotation_cast_bound_max:rotation_max,cast_action_component_bounds:actual_action,delta_eval_value:delta_eval.value,delta_eval_error:delta_eval.double_error,delta_subtraction_bounds:eta,delta_subtraction_propagated:eta_action,worst_budget_envelope_relative_l1:up(worst_envelope/norm),cast_identity_component_bounds:identity_bounds,cast_identity_mismatches:identity_mismatches,cast_identity_maximum_bound_ratio:identity_ratio})
    }
}
