//! Pure host authority over the FINAL builder DataSet. No filesystem/device API.
//! Directional/material/anchor Jacobians come from the rendered native helpers.
use crate::{data::{DataSet,PdrdRigidState},scene::Scene,restoring_curve::{self,BaseSnapshot,Target,MassEvidence}};
use nalgebra::{DMatrix,Matrix3,Vector3};
use serde::{Deserialize,Serialize};
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,HashSet},ffi::c_void};

type Result<T> = std::result::Result<T,String>;
fn bad(s:&str)->String{format!("RESTORING_MASS_INVALID: {s}")}
fn hash(b:&[u8])->String{format!("{:x}",Sha256::digest(b))}
fn digest_valid(s:&str)->bool{s.len()==64&&s.bytes().all(|v|v.is_ascii_digit()||(b'a'..=b'f').contains(&v))}
#[derive(Clone,Debug,Serialize)]
pub struct SolverIdentity{pub source_stamp:String,pub selected_backend:String,pub abi_version:u32,pub layout_sha256:String,pub kernel_table_sha256:String}
#[derive(Debug,Serialize)]
pub struct Response{pub kind:String,pub version:u32,pub scene_uuid:String,pub basis_sha256:String,pub request_sha256:String,pub solver_identity:SolverIdentity,pub authority_sha256:String,pub targets:Vec<ResponseRow>}
#[derive(Debug,Serialize)]
pub struct ResponseRow{pub target:Value,pub needs_mass:bool,pub native_kind:u32,pub target_index:u32,pub metric_eff:Option<f64>,pub metric_unit:Option<String>,pub quality:Option<String>,pub mobility_eigenvalues:Option<[f64;3]>,pub zero_direction_count:Option<u32>,pub anisotropy_ratio:Option<f64>}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]
struct Request{kind:String,version:u32,scene_uuid:String,basis_sha256:String,targets:Vec<RequestRow>}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]
struct RequestRow{target:RequestTarget,needs_mass:bool}
#[derive(Clone,Deserialize)]#[serde(tag="kind",deny_unknown_fields)]
enum RequestTarget{
    #[serde(rename="JOINT")]Joint{joint_uuid:String,channel:String},
    #[serde(rename="STITCH_ROW")]Stitch{row_index:u32,rows_sha256:String},
}
impl RequestTarget{
    fn key(&self)->String{match self{Self::Joint{joint_uuid,channel}=>format!("J/{joint_uuid}/{channel}"),Self::Stitch{row_index,..}=>format!("S/{row_index}")}}
    fn unresolved(&self)->Target{match self{Self::Joint{joint_uuid,channel}=>Target::Joint{joint_uuid:joint_uuid.clone(),channel:channel.clone()},Self::Stitch{row_index,rows_sha256}=>Target::Stitch{row_index:*row_index,rows_sha256:rows_sha256.clone(),epsilon_m:0.,rest_length_m:0.}}}
}
fn target_json(t:&Target)->Value{match t{Target::Joint{joint_uuid,channel}=>json!({"kind":"JOINT","joint_uuid":joint_uuid,"channel":channel}),Target::Stitch{row_index,rows_sha256,epsilon_m,rest_length_m}=>json!({"kind":"STITCH_ROW","row_index":row_index,"rows_sha256":rows_sha256,"epsilon_m":epsilon_m,"rest_length_m":rest_length_m})}}
extern "C"{
    fn restoring_mass_rest_states(vertex_properties:*const c_void,body_properties:*const c_void,positions:*const c_void,vertex_count:u32,vertex_list:*const u32,rest_centered:*const c_void,member_count:u32,body_count:u32,output:*mut c_void)->i32;
    fn restoring_mass_joint_jacobian(family:u32,channel:u32,record:*const c_void,xyz:*const f32,vertex_count:u32,states:*const c_void,body_count:u32,jacobian:*mut f32)->i32;
    fn restoring_mass_body_basis(joint_mode:u32,joint_axis:*const f32,has_t:u32,t_axis:*const f32,t_mode:u32,has_r:u32,r_axis:*const f32,r_mode:u32,fixed:*const f32,fixed_count:u32,basis:*mut f32,count:*mut u32)->i32;
}

struct Body{members:Vec<usize>,center:Vector3<f64>,inertia:Matrix3<f64>,mass:f64,state:PdrdRigidState,kinetic:DMatrix<f64>,forbidden:DMatrix<f64>,root_mode:u32,inverse:Result<DMatrix<f64>>}
struct Authority{positions:Vec<Vector3<f64>>,masses:Vec<f64>,fixed:Vec<bool>,body_index:Vec<Option<usize>>,bodies:Vec<Body>,ws:f64}
fn vec3(v:&crate::data::Vec3f)->Vector3<f64>{Vector3::new(v[0] as f64,v[1] as f64,v[2] as f64)}
fn skew(p:Vector3<f64>)->Matrix3<f64>{Matrix3::new(0.,-p[2],p[1],p[2],0.,-p[0],-p[1],p[0],0.)}
// Control events are strictly positive under the shared sealed root validator.
// LOCAL_REST at t=0 therefore uses the validated initial mode, not a future event.
fn initial_root_mode(mode:crate::joint_root_motion_wire::RootMotionMode)->u32{
    use crate::joint_root_motion_wire::RootMotionMode;
    match mode{RootMotionMode::Static=>1,RootMotionMode::Kinematic=>2,RootMotionMode::Dynamic=>3}
}
// Invert the restricted kinetic metric, NOT P M^-1 P. The forbidden basis is
// produced by the actual float native projector, including its rank cutoff.
fn constrained_inverse(m:&DMatrix<f64>,b:&DMatrix<f64>)->Result<DMatrix<f64>>{
    if !m.iter().chain(b.iter()).all(|x|x.is_finite()){return Err(bad("nonfinite restricted mass"))}
    let dim=m.nrows();if dim!=m.ncols()||b.ncols()!=dim{return Err(bad("restricted mass dimensions"))}
    if b.nrows()>dim{return Err(bad("constraint rank"))}
    let free=dim-b.nrows();if free==0{return Ok(DMatrix::zeros(dim,dim))}
    // B already has its native rank; SI coordinate conversion may remove
    // orthonormality. Extract exactly dim-rank null vectors, without a second
    // rank decision or an invalid I-B^T B shortcut.
    let n=if b.nrows()==0{DMatrix::identity(dim,dim)}else{
        let gram=b.transpose()*b;let basis=gram.symmetric_eigen();let mut order:Vec<_>=(0..dim).collect();order.sort_by(|&i,&j|basis.eigenvalues[i].total_cmp(&basis.eigenvalues[j]));
        let mut n=DMatrix::zeros(dim,free);for c in 0..free{n.set_column(c,&basis.eigenvectors.column(order[c]));}n
    };
    let restricted=n.transpose()*m*&n;
    if !restricted.iter().all(|v|v.is_finite()){return Err(bad("nonfinite restricted kinetic mass"))}
    let scale=restricted.norm();let tolerance=128.*f64::EPSILON*free as f64;
    if !scale.is_finite()||scale==0.||(&restricted-restricted.transpose()).norm()>tolerance*scale{return Err(bad("nonsymmetric restricted kinetic mass"))}
    // The positive definite kinetic metric is solved directly. No jitter,
    // eigenvalue clipping or spectral inverse of almost repeated small modes.
    let factor=restricted.clone().cholesky().ok_or_else(||bad("non-SPD restricted kinetic mass"))?;
    let inv=factor.inverse();
    if !inv.iter().all(|v|v.is_finite()){return Err(bad("nonfinite restricted inverse"))}
    // Normwise backward error, not a condition-number limit. Both product
    // terms are scaled separately to avoid rejecting a valid SI scale or
    // overflowing ||R||*||R^-1||. The dimension-scaled roundoff gate is fixed.
    let inv_scale=inv.norm();let residual=(&restricted*&inv-DMatrix::identity(free,free)).norm();
    let backward=(residual/scale)/inv_scale;
    if !inv_scale.is_finite()||inv_scale==0.||!backward.is_finite()||backward>tolerance{return Err(bad("restricted inverse backward residual"))}
    let w=&n*inv*n.transpose();if !w.iter().all(|v|v.is_finite()){return Err(bad("nonfinite constrained mobility"))}Ok(w)
}
fn root_mobility(mode:u32,m:&DMatrix<f64>,b:&DMatrix<f64>)->Result<DMatrix<f64>>{
    if mode==1||mode==2{Ok(DMatrix::zeros(m.nrows(),m.ncols()))}else{constrained_inverse(m,b)}
}
impl Authority{
    fn new(scene:&Scene,data:&DataSet,needs_mass:bool)->Result<Self>{
        let ws=scene.args().world_scaling as f64;if !(ws>0.)||!ws.is_finite(){return Err(bad("world scaling"))}
        let pos=data.vertex.curr.as_slice();let prop=data.prop.vertex.as_slice();
        if pos.len()!=prop.len()||pos.len()>u32::MAX as usize{return Err(bad("final vertex shape"))}
        let positions:Vec<_>=pos.iter().map(|p|vec3(p)/ws).collect();
        let masses:Vec<_>=prop.iter().map(|p|p.mass as f64/ws.powi(3)).collect();
        if !positions.iter().flat_map(|p|p.iter()).all(|x|x.is_finite())||!masses.iter().all(|x|x.is_finite()&&*x>=0.){return Err(bad("final mass/rest positions"))}
        let mut fixed=vec![false;pos.len()];for p in data.constraint.fix.as_slice(){let i=p.index as usize;if i>=fixed.len(){return Err(bad("fixed vertex index"))}fixed[i]=true;}
        let body_props=data.prop.pdrd_body.as_slice();let members_all=data.pdrd_vert_list.as_slice();let centered=data.pdrd_rest_centered.as_slice();
        if centered.len()!=members_all.len(){return Err(bad("actual rigid rest-centered shape"))}
        let mut native_states=vec![PdrdRigidState::default();body_props.len()];
        if !native_states.is_empty(){let code=unsafe{restoring_mass_rest_states(prop.as_ptr() as _,body_props.as_ptr() as _,pos.as_ptr() as _,pos.len() as u32,members_all.as_ptr(),centered.as_ptr() as _,members_all.len().try_into().map_err(|_|bad("rigid member budget"))?,body_props.len().try_into().map_err(|_|bad("rigid body budget"))?,native_states.as_mut_ptr() as _)};if code!=0{return Err(bad(&format!("actual rest fit refusal {code}")))}}
        let mut bodies=Vec::new();let mut body_index=vec![None;pos.len()];
        let mut root_modes=BTreeMap::new();if let Some(input)=&scene.joint_input{let owned=input.records();for root in &owned.roots{
            if owned.bodies.get(&root.body_uuid)!=Some(&root.runtime_body_index)||root.runtime_body_index as usize>=body_props.len()||root_modes.insert(root.runtime_body_index,initial_root_mode(root.mode)).is_some(){return Err(bad("actual prescribed-root identity/membership"))}
        }}
        for (bi,b) in data.prop.pdrd_body.as_slice().iter().enumerate(){
            let start=b.vertex_start as usize;let end=start.checked_add(b.vertex_count as usize).ok_or_else(||bad("body range overflow"))?;
            let slice=data.pdrd_vert_list.as_slice().get(start..end).ok_or_else(||bad("body membership range"))?;
            if slice.is_empty(){return Err(bad("empty rigid body"))}
            let mut members=Vec::new();for &v in slice{let v=v as usize;if v>=positions.len()||body_index[v].replace(bi).is_some(){return Err(bad("duplicate/outside rigid membership"))}if prop[v].pdrd_body_index!=bi as u32+1{return Err(bad("rigid membership mismatch"))}members.push(v);}
            let mass=members.iter().map(|&v|masses[v]).sum::<f64>();
            // Native fitted frame is relative to the actual builder centroid.
            let mut state=native_states[bi];for k in 0..3{state.anchor[k]=(state.anchor[k] as f64/ws) as f32;state.x[k]=(state.x[k] as f64/ws) as f32;}
            let origin=Vector3::from_column_slice(&state.anchor.map(f64::from))+Vector3::from_column_slice(&state.x.map(f64::from));
            let rotation=Matrix3::from_column_slice(&state.rotation.map(f64::from));
            let center=if mass>0.{members.iter().map(|&v|positions[v]*masses[v]).fold(Vector3::zeros(),|a,v|a+v)/mass}else{origin};
            let mut inertia=Matrix3::zeros();let mut kinetic=DMatrix::zeros(6,6);
            for (local,&v) in members.iter().enumerate(){
                let r=positions[v]-center;inertia+=(Matrix3::identity()*r.dot(&r)-r*r.transpose())*masses[v];
                let mut l=DMatrix::zeros(3,6);for k in 0..3{l[(k,k)]=1.;}let s=-skew(rotation*vec3(&centered[start+local])/ws);for r in 0..3{for c in 0..3{l[(r,3+c)]=s[(r,c)];}}kinetic+=l.transpose()*l*masses[v];
            }
            let restricted=(||->Result<(DMatrix<f64>,DMatrix<f64>)>{
                // Actual root tangent removes all six prescribed rigid DOFs;
                // these roots intentionally do not overlap constraint.fix.
                let mode=root_modes.get(&(bi as u32)).copied().unwrap_or(0);
                if mode==1||mode==2{return Ok((root_mobility(mode,&kinetic,&DMatrix::identity(6,6))?,DMatrix::identity(6,6)))}
                if !needs_mass{return Ok((DMatrix::zeros(6,6),DMatrix::zeros(0,6)))}
                let mut basis=[0f32;36];let mut count=0u32;
                let locks:Vec<_>=data.translation_lock.as_slice().iter().filter(|l|l.pdrd_body_index==bi as u32+1).collect();
                if locks.len()>1{return Err(bad("multiple rigid lock records"))}
                let lock=locks.first().copied();let zero=[0f32;3];
                let ta=lock.map(|l|[l.axis[0],l.axis[1],l.axis[2]]).unwrap_or(zero);
                let ra=lock.map(|l|[l.rotation_axis[0],l.rotation_axis[1],l.rotation_axis[2]]).unwrap_or(zero);
                // Mode carries all-axis enablement; a nonzero axis enables axis modes.
                let ht=lock.map(|l|l.translation_mode==crate::data::TRANSLATION_LOCK_ALL||l.axis.norm_squared()>0.).unwrap_or(false);
                let hr=lock.map(|l|l.rotation_mode==crate::data::ROTATION_LOCK_ALL||l.rotation_axis.norm_squared()>0.).unwrap_or(false);
                let fixed_xyz:Vec<f32>=members.iter().enumerate().filter(|(_,v)|fixed[**v]).flat_map(|(k,_)|(rotation*vec3(&centered[start+k])).map(|x|x as f32).as_slice().to_vec()).collect();
                let ja=[b.joint_axis[0],b.joint_axis[1],b.joint_axis[2]];
                let code=unsafe{restoring_mass_body_basis(b.joint_mode,ja.as_ptr(),ht as u32,ta.as_ptr(),lock.map(|l|l.translation_mode).unwrap_or(0),hr as u32,ra.as_ptr(),lock.map(|l|l.rotation_mode).unwrap_or(0),fixed_xyz.as_ptr(),(fixed_xyz.len()/3) as u32,basis.as_mut_ptr(),&mut count)};
                if code!=0||count>6{return Err(bad(&format!("actual rigid projector refusal {code}")))}
                let mut forbidden=DMatrix::<f64>::from_fn(count as usize,6,|r,c|basis[6*r+c] as f64);for r in 0..count as usize{for c in 0..3{forbidden[(r,c)]*=ws;}}
                Ok((root_mobility(mode,&kinetic,&forbidden)?,forbidden))
            })();
            let (inverse,forbidden)=match restricted{Ok((i,b))=>(Ok(i),b),Err(e)=>(Err(e),DMatrix::zeros(0,6))};
            state.mass_total=mass as f32;state.reference_inertia=inertia.map(|x|x as f32).as_slice().try_into().unwrap();
            bodies.push(Body{members,center,inertia,mass,state,kinetic,forbidden,root_mode:root_modes.get(&(bi as u32)).copied().unwrap_or(0),inverse});
        }
        for (i,p) in prop.iter().enumerate(){if (p.pdrd_body_index!=0)!=body_index[i].is_some(){return Err(bad("rigid final membership completeness"))}}
        Ok(Self{positions,masses,fixed,body_index,bodies,ws})
    }
    fn mobility(&self,data:&DataSet,nodal:&BTreeMap<usize,DMatrix<f64>>,rigid:&BTreeMap<usize,DMatrix<f64>>,rows:usize)->Result<DMatrix<f64>>{
        let mut out=DMatrix::zeros(rows,rows);
        for (&v,j) in nodal{
            if self.fixed[v]{continue}
            if data.translation_lock_index.as_slice().get(v).copied().unwrap_or(u32::MAX)!=u32::MAX{return Err(bad("AUTO aggregate material lock requires actual lockframe authority; not approximated"))}
            if self.masses[v]<=0.{return Err(bad("zero mobile nodal mass"))}out+=j*j.transpose()/self.masses[v];
        }
        for (&b,j) in rigid{let inverse=self.bodies[b].inverse.as_ref().map_err(Clone::clone)?;out+=j*inverse*j.transpose();}
        if !out.iter().all(|x|x.is_finite()){return Err(bad("nonfinite mobility"))}Ok(out)
    }
}

struct Resolved{target:Target,kind:u32,index:u32,rows:usize,nodal:BTreeMap<usize,DMatrix<f64>>,rigid:BTreeMap<usize,DMatrix<f64>>,jacobian:Vec<f32>}
fn add(map:&mut BTreeMap<usize,DMatrix<f64>>,key:usize,j:DMatrix<f64>){map.entry(key).and_modify(|v|*v+=&j).or_insert(j);}
fn bytes<'a>(base:&'a BaseSnapshot,name:&str)->Result<&'a [u8]>{base.bytes.get(name).map(Vec::as_slice).ok_or_else(||bad(&format!("sealed input absent: {name}")))}
fn resolve(scene:&Scene,data:&DataSet,base:&BaseSnapshot,a:&Authority,t:&Target)->Result<Resolved>{
    let mut nodal=BTreeMap::new();let mut rigid=BTreeMap::new();
    if let Target::Stitch{row_index,rows_sha256,..}=t{
        let stitches=data.constraint.stitch.as_slice();let row=stitches.get(*row_index as usize).ok_or_else(||bad("stitch row index"))?;
        let ind=bytes(base,"bin/stitch_ind.bin")?;let w=bytes(base,"bin/stitch_w.bin")?;
        if restoring_curve::rows_hash(ind,w,stitches.len() as u64)?!=*rows_sha256{return Err(bad("ordered stitch source hash"))}
        for (i,s) in stitches.iter().enumerate(){for k in 0..6{let p=6*i+k;
            let sealed_index=u64::from_le_bytes(ind[8*p..8*p+8].try_into().unwrap());
            let sealed_weight=u32::from_le_bytes(w[4*p..4*p+4].try_into().unwrap());
            if sealed_index!=s.index[k] as u64||sealed_weight!=s.weight[k].to_bits(){return Err(bad("final stitch differs from sealed source order/weights"))}
        }}
        let params=data.param_arrays.vertex.as_slice();let props=data.prop.vertex.as_slice();
        let mut gap=0.;let mut low=Vector3::repeat(f64::INFINITY);let mut high=Vector3::repeat(f64::NEG_INFINITY);let mut coefficients:BTreeMap<usize,f64>=BTreeMap::new();
        let mut sums=[0.;2];for k in 0..6{let v=row.index[k] as usize;if v>=a.positions.len(){return Err(bad("stitch vertex"))}let weight=row.weight[k] as f64;if !weight.is_finite(){return Err(bad("stitch weight"))}sums[k/3]+=weight;
            let pi=props[v].param_index as usize;let g=params.get(pi).ok_or_else(||bad("stitch gap param index"))?.ghat as f64/a.ws;
            if !g.is_finite()||g<0.{return Err(bad("stitch contact gap"))}gap+=weight*g/2.;for r in 0..3{low[r]=low[r].min(a.positions[v][r]);high[r]=high[r].max(a.positions[v][r]);}*coefficients.entry(v).or_default()+=if k<3{weight}else{-weight};}
        if sums.iter().any(|s|(s-1.).abs()>2e-6)||!gap.is_finite()||gap<0.{return Err(bad("stitch affine weights/rest gap"))}
        let l=(high-low).norm();let native_gap=gap*a.ws;let native_l=l*a.ws;
        let native_eps=1e-9f64.max(1e-4*native_gap).max(32.*f32::EPSILON as f64*native_l);let eps=native_eps/a.ws;
        if !eps.is_finite()||!(native_eps as f32>0.)||native_eps>1e-6f64.max(0.01*native_gap.max(native_l)){return Err(bad("stitch smoothing outside frozen contract"))}
        let rest=gap.hypot(eps);
        for (v,c) in coefficients{if c==0.{continue}if a.body_index[v].is_some(){return Err(bad("stitch PDRD endpoint is unsupported by actual nodal assembler"))}add(&mut nodal,v,DMatrix::identity(3,3)*c);}
        // Contract stitch mass is sum aggregated coefficient^2/m, a scalar
        // nodal estimate, rather than the coordinate's instantaneous radial J.
        return Ok(Resolved{target:Target::Stitch{row_index:*row_index,rows_sha256:rows_sha256.clone(),epsilon_m:eps,rest_length_m:rest},kind:4,index:*row_index,rows:3,nodal,rigid,jacobian:Vec::new()});
    }
    let Target::Joint{joint_uuid,channel}=t else{unreachable!()};
    if crate::joint::canonical_uuid(joint_uuid).as_deref()!=Some(joint_uuid.as_str()){return Err(bad("canonical joint UUID"))}
    let ch=match channel.as_str(){"twist_x"=>0,"swing_y"=>1,"swing_z"=>2,"anchor"=>3,_=>return Err(bad("channel"))};let rows=if ch==3{3}else{1};
    let input=scene.joint_input.as_ref().ok_or_else(||bad("joint carrier absent"))?;
    let owned=input.records();let soft=crate::soft_joint::prepare(input).map_err(|e|e.to_string())?;
    let rigid_id=owned.joints.iter().position(|j|j.wire.joint_uuid==*joint_uuid);
    let soft_id=soft.as_ref().and_then(|s|s.wires.iter().position(|j|j.wire.joint_uuid==*joint_uuid));
    if rigid_id.is_some()&&soft_id.is_some(){return Err(bad("ambiguous rigid/material target"))}
    let mut js=[0f32;72];let xyz:Vec<f32>=a.positions.iter().flat_map(|p|p.iter().map(|x|*x as f32)).collect();let states:Vec<_>=a.bodies.iter().map(|b|b.state).collect();
    let (kind,index)=if let Some(i)=rigid_id{
        let joint=&owned.joints[i];if !joint.wire.enabled{return Err(bad("disabled joint target"))}
        if ch<3&&(joint.wire.angle_stiffness_nm_per_rad!=0.||joint.wire.angle_damping_nms_per_rad!=0.||joint.wire.max_angle_rad!=0.||joint.wire.angular_hard_hinge_v1.is_some()||joint.wire.joint_d6_hard_v1.is_some()){return Err("RESTORING_CONFLICT: angular V2/hard owner".into())}
        let family=if ch==3{2}else{0};
        let code=if ch==3{let records=crate::joint_gpu::from_validated(owned)?;unsafe{restoring_mass_joint_jacobian(family,ch,&records[i] as *const _ as _,xyz.as_ptr(),a.positions.len() as u32,states.as_ptr() as _,states.len() as u32,js.as_mut_ptr())}}
            else{let records=crate::joint_angular_control_gpu::from_validated(owned)?;let record=records.get(i).ok_or_else(||bad("actual directional chart record absent"))?;if record.enabled!=1{return Err(bad("directional chart not enabled"))}unsafe{restoring_mass_joint_jacobian(family,ch,record as *const _ as _,xyz.as_ptr(),a.positions.len() as u32,states.as_ptr() as _,states.len() as u32,js.as_mut_ptr())}};
        if code!=0{return Err(bad(&format!("actual native rest Jacobian refusal {code}")))}
        for (side,body) in [joint.parent_runtime_body_index,joint.child_runtime_body_index].into_iter().enumerate(){if body==u32::MAX{continue}if body as usize>=a.bodies.len(){return Err(bad("joint body index"))}let mut j=DMatrix::zeros(rows,6);for r in 0..rows{for k in 0..6{j[(r,k)]=js[24*r+12*side+k] as f64;}}add(&mut rigid,body as usize,j);}
        (family,i as u32)
    }else if let Some(i)=soft_id{
        let s=soft.as_ref().unwrap();let record=&s.records[i];if record.enabled!=1||(ch<3&&record.control.enabled!=1){return Err(bad("disabled material chart target"))}
        let wire=&s.wires[i].wire;if ch<3&&(wire.angle_stiffness_nm_per_rad!=0.||wire.angle_damping_nms_per_rad!=0.||wire.max_angle_rad!=0.||wire.angular_hard_hinge_v1.is_some()||wire.joint_d6_hard_v1.is_some()){return Err("RESTORING_CONFLICT: material angular V2/hard owner".into())}
        let family=if ch==3{3}else{1};let code=unsafe{restoring_mass_joint_jacobian(family,ch,record as *const _ as _,xyz.as_ptr(),a.positions.len() as u32,states.as_ptr() as _,states.len() as u32,js.as_mut_ptr())};if code!=0{return Err(bad(&format!("actual FEM rest Jacobian refusal {code}")))}
        for (side,e) in [record.a,record.b].into_iter().enumerate(){if e.kind==0{continue}if e.kind==1{if e.id as usize>=a.bodies.len(){return Err(bad("material rigid body index"))}let mut j=DMatrix::zeros(rows,6);for r in 0..rows{for k in 0..6{j[(r,k)]=js[24*r+12*side+k] as f64;}}add(&mut rigid,e.id as usize,j);}else if e.kind==2{for v in 0..4{let id=e.vertices[v] as usize;if id>=a.positions.len()||a.body_index[id].is_some(){return Err(bad("material node membership"))}let mut j=DMatrix::zeros(rows,3);for r in 0..rows{for k in 0..3{j[(r,k)]=js[24*r+12*side+3*v+k] as f64;}}add(&mut nodal,id,j);}}else{return Err(bad("material endpoint family"))}}
        (family,i as u32)
    }else{return Err(bad("joint target absent"))};
    Ok(Resolved{target:t.clone(),kind,index,rows,nodal,rigid,jacobian:js.to_vec()})
}

fn evidence(a:&Authority,data:&DataSet,r:&Resolved)->Result<MassEvidence>{
    let mobility=a.mobility(data,&r.nodal,&r.rigid,r.rows)?;
    let anchor=r.kind==2||r.kind==3;let quality=match r.kind{0=>"LOCAL_RIGID_ESTIMATE",1|3=>"FEM_LOCAL_MASS_ESTIMATE",2=>"ANCHOR_MEAN_COMPLIANCE_ESTIMATE",4=>"STITCH_NODAL_LOCAL_ESTIMATE",_=>return Err(bad("native family"))}.to_string();
    if anchor{
        let eig=mobility.symmetric_eigen();let mut values:[f64;3]=eig.eigenvalues.as_slice().try_into().unwrap();values.sort_by(f64::total_cmp);let max=values[2];for v in &mut values{if *v < -max*1e-8{return Err(bad("negative anchor mobility"))}if v.abs()<=max*1e-10{*v=0.;}}
        let sum=values.iter().sum::<f64>();if !(sum>0.)||!sum.is_finite(){return Err(bad("all-fixed anchor mobility"))}let min=values.iter().copied().find(|v|*v>0.).unwrap();
        Ok(MassEvidence{metric_eff:3./sum,quality,mobility_eigenvalues:Some(values),zero_direction_count:Some(values.iter().filter(|v|**v==0.).count() as u32),anisotropy_ratio:Some(max/min)})
    }else{let mu=mobility[(0,0)];if !(mu>0.)||!mu.is_finite(){return Err(bad("all-fixed/zero target mobility"))}Ok(MassEvidence{metric_eff:1./mu,quality,mobility_eigenvalues:None,zero_direction_count:None,anisotropy_ratio:None})}
}
pub fn target_geometry(scene:&Scene,data:&DataSet,base:&BaseSnapshot,target:&Target)->Result<Target>{let a=Authority::new(scene,data,false)?;Ok(resolve(scene,data,base,&a,target)?.target)}
pub fn target_slot(scene:&Scene,data:&DataSet,base:&BaseSnapshot,target:&Target)->Result<(u32,u32)>{let a=Authority::new(scene,data,false)?;let r=resolve(scene,data,base,&a,target)?;Ok((r.kind,r.index))}
pub fn recompute(scene:&Scene,data:&DataSet,base:&BaseSnapshot,target:&Target)->Result<MassEvidence>{
    let a=Authority::new(scene,data,true)?;let r=resolve(scene,data,base,&a,target)?;
    if let(Target::Stitch{epsilon_m:e,rest_length_m:l,..},Target::Stitch{epsilon_m:ae,rest_length_m:al,..})=(target,&r.target){if (*e-*ae).abs()>ae.abs()*1e-6||(*l-*al).abs()>al.abs()*1e-6{return Err(bad("sealed stitch geometry differs from final builder"))}}
    evidence(&a,data,&r)
}

// V1 canonical LE projection. Explicit scalars only; no repr(C) padding,
// addresses, platform usize, serde object order, or response diagnostics.
struct Projection(Vec<u8>);
impl Projection{fn u32(&mut self,v:u32){self.0.extend(v.to_le_bytes())}fn f64(&mut self,v:f64){self.0.extend(v.to_bits().to_le_bytes())}fn f32(&mut self,v:f32){self.0.extend(v.to_bits().to_le_bytes())}fn count(&mut self,n:usize)->Result<()>{self.u32(n.try_into().map_err(|_|bad("projection count"))?);Ok(())}fn string(&mut self,s:&str)->Result<()>{self.count(s.len())?;self.0.extend(s.as_bytes());Ok(())}}
fn projection(a:&Authority,data:&DataSet,resolved:&[Resolved])->Result<Vec<u8>>{
    let mut p=Projection(b"ppf.restoring_curves.mass_authority/le-v1\0".to_vec());p.f64(a.ws);p.count(a.positions.len())?;
    for i in 0..a.positions.len(){for k in 0..3{p.f64(a.positions[i][k])}p.f64(a.masses[i]);p.u32(a.fixed[i] as u32);p.u32(a.body_index[i].map(|v|v as u32).unwrap_or(u32::MAX));p.u32(data.translation_lock_index.as_slice().get(i).copied().unwrap_or(u32::MAX));}
    p.count(a.bodies.len())?;for b in &a.bodies{p.count(b.members.len())?;for &v in &b.members{p.u32(v as u32)}p.f64(b.mass);for k in 0..3{p.f64(b.center[k])}for r in 0..3{for c in 0..3{p.f64(b.inertia[(r,c)])}}for k in 0..3{p.f32(b.state.anchor[k]);p.f32(b.state.x[k])}for &v in &b.state.rotation{p.f32(v)}for r in 0..6{for c in 0..6{p.f64(b.kinetic[(r,c)])}}p.u32(b.root_mode);p.count(b.forbidden.nrows())?;for r in 0..b.forbidden.nrows(){for c in 0..6{p.f64(b.forbidden[(r,c)])}}}
    let locks=data.translation_lock.as_slice();p.count(locks.len())?;for l in locks{for k in 0..3{p.f32(l.axis[k])}p.u32(l.translation_mode);p.f32(l.total_mass);p.u32(l.pdrd_body_index);p.u32(l.dmap_index);for k in 0..3{p.f32(l.rotation_axis[k])}p.u32(l.rotation_mode);for k in 0..3{p.f32(l.anchor[k])}}
    p.count(data.prop.pdrd_body.as_slice().len())?;for b in data.prop.pdrd_body.as_slice(){p.u32(b.joint_mode);for k in 0..3{p.f32(b.joint_axis[k])}for k in 0..3{p.f32(b.rest_centroid[k])}}
    let props=data.prop.vertex.as_slice();let params=data.param_arrays.vertex.as_slice();let stitches=data.constraint.stitch.as_slice();p.count(stitches.len())?;
    for row in stitches{for k in 0..6{p.u32(row.index[k]);p.f32(row.weight[k]);let pi=props.get(row.index[k] as usize).ok_or_else(||bad("projection stitch index"))?.param_index as usize;p.f32(params.get(pi).ok_or_else(||bad("projection gap index"))?.ghat);}}
    p.count(resolved.len())?;for r in resolved{p.u32(r.kind);p.u32(r.index);match &r.target{Target::Joint{joint_uuid,channel}=>{p.string(joint_uuid)?;p.string(channel)?},Target::Stitch{rows_sha256,epsilon_m,rest_length_m,..}=>{p.string(rows_sha256)?;p.f64(*epsilon_m);p.f64(*rest_length_m)}}p.count(r.jacobian.len())?;for &j in &r.jacobian{p.f32(j)}p.count(r.nodal.len())?;for(&v,j)in &r.nodal{p.u32(v as u32);for rr in 0..r.rows{for c in 0..3{p.f64(j[(rr,c)])}}}p.count(r.rigid.len())?;for(&b,j)in &r.rigid{p.u32(b as u32);for rr in 0..r.rows{for c in 0..6{p.f64(j[(rr,c)])}}}}
    Ok(p.0)
}
pub fn query(scene:&Scene,data:&DataSet,base:&BaseSnapshot,request:&[u8],identity:SolverIdentity)->Result<Response>{
    if request.len()>16*1024*1024{return Err(bad("request byte budget"))}
    let strict:crate::joint::StrictValue=serde_json::from_slice(request).map_err(|e|bad(&e.to_string()))?;
    let req:Request=serde_json::from_value(strict.0).map_err(|e|bad(&e.to_string()))?;
    if req.kind!="ppf.restoring_curves.mass_request"||req.version!=1||req.targets.is_empty()||req.targets.len()>65536{return Err(bad("request kind/version/target budget"))}
    if req.scene_uuid!=base.context.scene_uuid||req.basis_sha256!=base.context.scene_inputs_sha256{return Err("RESTORING_BASIS_MISMATCH: query context".into())}
    if !digest_valid(&identity.layout_sha256)||!digest_valid(&identity.kernel_table_sha256)||identity.source_stamp.is_empty()||!matches!(identity.selected_backend.as_str(),"cuda"|"cpu")||identity.abi_version==0{return Err(bad("compiled solver identity"))}
    let mut seen=HashSet::new();for row in &req.targets{if !seen.insert(row.target.key()){return Err(bad("duplicate underlying target"))}}
    let a=Authority::new(scene,data,req.targets.iter().any(|r|r.needs_mass))?;
    let mut targets=Vec::new();let mut resolved=Vec::new();
    for row in &req.targets{let r=resolve(scene,data,base,&a,&row.target.unresolved())?;let m=if row.needs_mass{Some(evidence(&a,data,&r)?)}else{None};
        targets.push(ResponseRow{target:target_json(&r.target),needs_mass:row.needs_mass,native_kind:r.kind,target_index:r.index,metric_eff:m.as_ref().map(|v|v.metric_eff),metric_unit:m.as_ref().map(|_|if r.kind<=1{"kg_m2"}else{"kg"}.into()),quality:m.as_ref().map(|v|v.quality.clone()),mobility_eigenvalues:m.as_ref().and_then(|v|v.mobility_eigenvalues),zero_direction_count:m.as_ref().and_then(|v|v.zero_direction_count),anisotropy_ratio:m.as_ref().and_then(|v|v.anisotropy_ratio)});resolved.push(r);
    }
    Ok(Response{kind:"ppf.restoring_curves.mass_authority".into(),version:1,scene_uuid:req.scene_uuid,basis_sha256:req.basis_sha256,request_sha256:hash(request),solver_identity:identity,authority_sha256:hash(&projection(&a,data,&resolved)?),targets})
}

/// Test-only raw observables. NOT part of the strict query wire or a capability.
/// File creation belongs to the explicit cfg(test) harness below.
#[cfg(test)]
fn diagnostic_inverse(b:&Body)->Value{
    fn rows(m:&DMatrix<f64>)->Vec<Vec<f64>>{(0..m.nrows()).map(|r|(0..m.ncols()).map(|c|m[(r,c)]).collect()).collect()}
    // Actual W is the value already stored by Authority::new. Every other
    // factor below is explicitly a diagnostic replay, never substituted for W.
    let observed=b.inverse.as_ref().ok().map(rows);
    let dim=b.kinetic.nrows();let free=dim-b.forbidden.nrows();
    if free==0{return json!({"actual_inverse_row_major":observed,"replay_scope":"TEST_ONLY_SPECTRAL_REPLAY","free_dimension":0,"prescribed_root_mode":b.root_mode})}
    let gram=b.forbidden.transpose()*&b.forbidden;
    let basis=gram.symmetric_eigen();let mut order:Vec<_>=(0..dim).collect();order.sort_by(|&i,&j|basis.eigenvalues[i].total_cmp(&basis.eigenvalues[j]));
    let mut n=DMatrix::zeros(dim,free);for c in 0..free{n.set_column(c,&basis.eigenvectors.column(order[c]));}
    let restricted=n.transpose()*&b.kinetic*&n;let eig=restricted.clone().symmetric_eigen();
    let reconstructed=&eig.eigenvectors*DMatrix::from_diagonal(&eig.eigenvalues)*eig.eigenvectors.transpose();
    let orthogonal=eig.eigenvectors.transpose()*&eig.eigenvectors-DMatrix::identity(free,free);
    let eig_residual=&restricted*&eig.eigenvectors-&eig.eigenvectors*DMatrix::from_diagonal(&eig.eigenvalues);
    let mut replay_inv=DMatrix::zeros(free,free);for k in 0..free{let u=eig.eigenvectors.column(k);replay_inv+=u*u.transpose()/eig.eigenvalues[k];}
    let replay_w=&n*&replay_inv*n.transpose();
    let replay_residual=&restricted*&replay_inv-DMatrix::identity(free,free);
    let actual_residual=b.inverse.as_ref().ok().map(|w|rows(&(&restricted*(n.transpose()*w*&n)-DMatrix::identity(free,free))));
    json!({"actual_inverse_row_major":observed,"actual_restricted_inverse_residual":actual_residual,"replay_scope":"TEST_ONLY_SPECTRAL_REPLAY","free_dimension":free,"prescribed_root_mode":b.root_mode,
        "replay_N":rows(&n),"replay_restricted_M":rows(&restricted),"replay_eigenvalues":eig.eigenvalues.as_slice(),"replay_eigenvectors":rows(&eig.eigenvectors),
        "replay_eigen_orthogonality_residual":rows(&orthogonal),"replay_eigen_reconstruction_residual":rows(&(&reconstructed-&restricted)),"replay_eigen_equation_residual":rows(&eig_residual),
        "replay_restricted_inverse":rows(&replay_inv),"replay_inverse_residual":rows(&replay_residual),"replay_W":rows(&replay_w)})
}
#[cfg(test)]
pub fn diagnostic_snapshot(scene:&Scene,data:&DataSet,base:&BaseSnapshot,targets:&[Target])->Result<Value>{
    let a=Authority::new(scene,data,true)?;
    let density=base.bytes.get("bin/param/tet-density.bin").map(|b|{
        if b.len()!=4*data.prop.tet.as_slice().len(){return Err(bad("retained tet-density shape"))}
        Ok(b.chunks_exact(4).map(|p|f32::from_le_bytes(p.try_into().unwrap())).collect::<Vec<_>>())
    }).transpose()?;
    let mut rows=Vec::new();
    for t in targets{
        let r=resolve(scene,data,base,&a,t)?;
        let source_endpoints=if r.kind==0||r.kind==2{
            let j=&scene.joint_input.as_ref().unwrap().records().joints[r.index as usize];
            json!({"joint_uuid":j.wire.joint_uuid,"endpoints":[{"kind":if j.parent_runtime_body_index==u32::MAX{0}else{1},"id":j.parent_runtime_body_index,"vertices":[]},{"kind":if j.child_runtime_body_index==u32::MAX{0}else{1},"id":j.child_runtime_body_index,"vertices":[]}]})
        }else if r.kind==1||r.kind==3{
            let soft=crate::soft_joint::prepare(scene.joint_input.as_ref().unwrap()).map_err(|e|e.to_string())?.unwrap();let j=&soft.records[r.index as usize];
            json!({"joint_uuid":soft.wires[r.index as usize].wire.joint_uuid,"endpoints":[{"kind":j.a.kind,"id":j.a.id,"vertices":j.a.vertices},{"kind":j.b.kind,"id":j.b.id,"vertices":j.b.vertices}]})
        }else{Value::Null};
        let ev=evidence(&a,data,&r);
        let nodal:Vec<_>=r.nodal.iter().map(|(&id,j)|json!({"vertex":id,"j":(0..r.rows).map(|rr|(0..3).map(|c|j[(rr,c)]).collect::<Vec<_>>()).collect::<Vec<_>>()})).collect();
        let rigid:Vec<_>=r.rigid.iter().map(|(&id,j)|json!({"body":id,"j":(0..r.rows).map(|rr|(0..6).map(|c|j[(rr,c)]).collect::<Vec<_>>()).collect::<Vec<_>>()})).collect();
        let mass=match ev{Ok(m)=>json!({"metric_eff":m.metric_eff,"quality":m.quality,"mobility_eigenvalues":m.mobility_eigenvalues,"zero_direction_count":m.zero_direction_count,"anisotropy_ratio":m.anisotropy_ratio}),Err(e)=>json!({"refusal":e})};
        rows.push(json!({"target":target_json(&r.target),"native_kind":r.kind,"target_index":r.index,"rows":r.rows,"source_endpoint_mapping":source_endpoints,"native_jacobian_3x24":r.jacobian,"nodal_mapping":nodal,"rigid_mapping":rigid,"mass":mass}));
    }
    let bodies:Vec<_>=a.bodies.iter().enumerate().map(|(i,b)|{
        let bp=&data.prop.pdrd_body.as_slice()[i];let start=bp.vertex_start as usize;
        let centered:Vec<_>=data.pdrd_rest_centered.as_slice()[start..start+bp.vertex_count as usize].iter().map(|v|vec![v[0],v[1],v[2]]).collect();
        json!({"members":b.members,"rest_centered_native":centered,"state_anchor_si":b.state.anchor,"state_x_si":b.state.x,"rotation_column_major":b.state.rotation,"mass_si":b.mass,"com_si":b.center.as_slice(),"inertia_si_column_major":b.inertia.as_slice(),"kinetic_m_row_major":(0..6).map(|r|(0..6).map(|c|b.kinetic[(r,c)]).collect::<Vec<_>>()).collect::<Vec<_>>(),"initial_root_mode":b.root_mode,"native_rank":b.forbidden.nrows(),"forbidden_si":(0..b.forbidden.nrows()).map(|r|(0..6).map(|c|b.forbidden[(r,c)]).collect::<Vec<_>>()).collect::<Vec<_>>(),"joint_mode":bp.joint_mode,"joint_axis":bp.joint_axis.as_slice(),"inverse_refusal":b.inverse.as_ref().err(),"inverse_diagnostic":diagnostic_inverse(b)})
    }).collect();
    Ok(json!({"kind":"ppf.restoring_curves.test_only_raw_dataset","scene_uuid":base.context.scene_uuid,"basis_sha256":base.context.scene_inputs_sha256,"world_scaling":a.ws,
        "positions_native":data.vertex.curr.as_slice().iter().map(|v|vec![v[0],v[1],v[2]]).collect::<Vec<_>>(),
        "final_nodal_masses_native":data.prop.vertex.as_slice().iter().map(|v|v.mass).collect::<Vec<_>>(),"fixed":a.fixed,"body_index":a.body_index,"bodies":bodies,"locks":data.translation_lock.as_slice(),"lock_index":data.translation_lock_index.as_slice(),"targets":rows,
        "tets":data.mesh.mesh.tet.as_slice().iter().map(|v|vec![v[0],v[1],v[2],v[3]]).collect::<Vec<_>>(),"tet_properties":data.prop.tet.as_slice(),"retained_sealed_tet_density":density,
        "faces":data.mesh.mesh.face.as_slice().iter().map(|v|vec![v[0],v[1],v[2]]).collect::<Vec<_>>(),"face_properties":data.prop.face.as_slice(),
        "edges":data.mesh.mesh.edge.as_slice().iter().map(|v|vec![v[0],v[1]]).collect::<Vec<_>>(),"edge_properties":data.prop.edge.as_slice(),
        "pdrd_body_properties":data.prop.pdrd_body.as_slice(),"shell_face_count":data.shell_face_count,"rod_count":data.rod_count,"include_face_mass":scene.args().include_face_mass,
        "stitches":data.constraint.stitch.as_slice().iter().map(|s|json!({"indices":s.index.as_slice(),"weights":s.weight.as_slice()})).collect::<Vec<_>>(),
        "actual_builder_dataset":serde_json::to_value(data).map_err(|e|bad(&e.to_string()))?}))
}

#[cfg(test)]mod tests{
    use super::*;
    fn captured_sphere_metric()->DMatrix<f64>{DMatrix::from_row_slice(6,6,&[
        0.02246399546856992,0.,0.,0.,-4.233826729146701e-13,0.,
        0.,0.02246399546856992,0.,4.233826729146701e-13,0.,-1.477011593150373e-13,
        0.,0.,0.02246399546856992,0.,1.477011593150373e-13,0.,
        0.,4.233826729146701e-13,0.,2.326796644241177e-6,-3.0717337132625268e-15,3.832775376081366e-14,
        -4.233826729146701e-13,0.,1.477011593150373e-13,-3.0717337132625268e-15,1.8160358617544404e-6,-4.386073054250975e-15,
        0.,-1.477011593150373e-13,0.,3.832775376081366e-14,-4.386073054250975e-15,2.3267964636894237e-6])}
    #[test]fn inverse_repair_captured_sphere_and_si_scales(){
        let m=captured_sphere_metric();let b=DMatrix::zeros(0,6);
        for scale in [1e-12,1.,1e12]{let scaled=&m*scale;let w=constrained_inverse(&scaled,&b).unwrap();assert!((&scaled*&w-DMatrix::identity(6,6)).norm()<1e-11);assert!((w[(3,3)]*scale-429775.4178367933).abs()<1e-5);assert!((w[(4,4)]*scale-550649.9189029877).abs()<1e-5);}
    }
    #[test]fn inverse_repair_anisotropic_native_rank_and_nonunit_basis(){
        let m=DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1e-8,1.,1.00000000001,10000.]));
        let b=DMatrix::from_row_slice(1,4,&[0.,2.,2.,0.]);let w=constrained_inverse(&m,&b).unwrap();
        assert!((&b*&w).norm()<1e-10);
        // Explicit independent orthonormal tangent basis; no eigensolver oracle.
        let s=0.5f64.sqrt();let n=DMatrix::from_row_slice(4,3,&[1.,0.,0.,0.,s,0.,0.,-s,0.,0.,0.,1.]);
        let expected=&n*(n.transpose()*&m*&n).try_inverse().unwrap()*n.transpose();assert!((&w-expected).norm()/w.norm()<1e-12);
        assert_eq!(root_mobility(1,&m,&b).unwrap(),DMatrix::zeros(4,4));assert_eq!(root_mobility(2,&m,&b).unwrap(),DMatrix::zeros(4,4));
    }
    #[test]fn inverse_repair_rejects_nonspd_nonfinite_nonsymmetric(){
        let b=DMatrix::zeros(0,2);
        for m in [DMatrix::zeros(2,2),DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.,-1.])),DMatrix::from_row_slice(2,2,&[1.,0.1,0.,1.]),DMatrix::from_row_slice(2,2,&[f64::NAN,0.,0.,1.])]{assert!(constrained_inverse(&m,&b).is_err());}
        assert!(constrained_inverse(&DMatrix::identity(2,3),&b).is_err());
    }
    #[test]fn constrained_mass_is_not_euclidean_filtered_inverse(){let m=DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.,9.]));let s=0.5f64.sqrt();let b=DMatrix::from_row_slice(1,2,&[s,s]);let inv=constrained_inverse(&m,&b).unwrap();assert!((inv[(0,0)]-0.1).abs()<1e-12);assert!((inv[(0,1)]+0.1).abs()<1e-12);}
    #[test]fn all_fixed_inverse_zero(){let inv=constrained_inverse(&DMatrix::identity(6,6),&DMatrix::identity(6,6)).unwrap();assert_eq!(inv,DMatrix::zeros(6,6));}
    #[test]fn validated_root_initial_mode_tangent(){use crate::joint_root_motion_wire::RootMotionMode as M;assert_eq!(initial_root_mode(M::Static),1);assert_eq!(initial_root_mode(M::Kinematic),2);assert_eq!(initial_root_mode(M::Dynamic),3);}
    #[test]fn strict_false_bool_and_duplicate_json(){assert!(serde_json::from_str::<RequestRow>(r#"{"target":{"kind":"STITCH_ROW","row_index":0,"rows_sha256":"a"},"needs_mass":0}"#).is_err());assert!(serde_json::from_str::<crate::joint::StrictValue>(r#"{"x":1,"x":2}"#).is_err());}
    #[test]fn projection_scalars_are_fixed_little_endian(){let mut p=Projection(Vec::new());p.u32(0x01020304);p.f64(1.);p.f32(-0.);p.string("ab").unwrap();assert_eq!(p.0,vec![4,3,2,1,0,0,0,0,0,0,240,63,0,0,0,128,2,0,0,0,97,98]);}
    #[test]fn actual_builder_diagnostic_gate(){
        // Explicit environment is required. Ordinary cargo test never reads an
        // unrelated user's scene or writes diagnostic artifacts.
        let Ok(session)=std::env::var("PPF_MASS_QA_SESSION")else{eprintln!("NOT_RUN actual builder gate: PPF_MASS_QA_SESSION absent");return};
        let out=std::path::PathBuf::from(std::env::var("PPF_MASS_QA_OUTPUT").expect("fresh test output required"));
        let scope=std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../coordination/windows-resume-20261004/restoring-curve/native-mass").canonicalize().unwrap();
        let parent=out.parent().unwrap().canonicalize().expect("output parent must already exist");assert!(parent.starts_with(&scope),"test writes confined to owned native-mass directory");assert!(!out.exists(),"fresh output only");std::fs::create_dir(&out).unwrap();
        use clap::Parser;
        let args=crate::args::ProgramArgs::parse_from(["mass-host-qa","--path",session.as_str(),"--output",out.to_str().unwrap()]);
        let context_path=std::path::Path::new(&session).join("bin/pdrd_scene_context.cbor");
        let context_before=std::fs::read(&context_path).unwrap();
        let mut scene=Scene::new(&args);let base=restoring_curve::load_base_context(std::path::Path::new(&session)).unwrap();let data=crate::fresh_authority_dataset(&mut scene);
        let mut targets=Vec::new();if let Some(input)=&scene.joint_input{
            for j in &input.records().joints{if j.wire.enabled{targets.push(Target::Joint{joint_uuid:j.wire.joint_uuid.clone(),channel:"anchor".into()});if j.wire.angular_control_v1.is_some()&&j.wire.angle_stiffness_nm_per_rad==0.&&j.wire.angle_damping_nms_per_rad==0.&&j.wire.max_angle_rad==0.&&j.wire.angular_hard_hinge_v1.is_none()&&j.wire.joint_d6_hard_v1.is_none(){for c in ["twist_x","swing_y","swing_z"]{targets.push(Target::Joint{joint_uuid:j.wire.joint_uuid.clone(),channel:c.into()});}}}}
            if let Some(soft)=crate::soft_joint::prepare(input).unwrap(){for j in &soft.wires{if j.wire.enabled{targets.push(Target::Joint{joint_uuid:j.wire.joint_uuid.clone(),channel:"anchor".into()});if j.wire.angular_control_v1.is_some()&&j.wire.angle_stiffness_nm_per_rad==0.&&j.wire.angle_damping_nms_per_rad==0.&&j.wire.max_angle_rad==0.{for c in ["twist_x","swing_y","swing_z"]{targets.push(Target::Joint{joint_uuid:j.wire.joint_uuid.clone(),channel:c.into()});}}}}}
        }
        if !data.constraint.stitch.as_slice().is_empty(){let sha=restoring_curve::rows_hash(bytes(&base,"bin/stitch_ind.bin").unwrap(),bytes(&base,"bin/stitch_w.bin").unwrap(),data.constraint.stitch.size as u64).unwrap();for row_index in 0..data.constraint.stitch.size{targets.push(Target::Stitch{row_index,rows_sha256:sha.clone(),epsilon_m:0.,rest_length_m:0.});}}
        assert!(!targets.is_empty(),"actual fixture must contain supported targets");
        let raw=diagnostic_snapshot(&scene,&data,&base,&targets).unwrap();let mut queries=Vec::new();
        for row in raw["targets"].as_array().unwrap(){let mut t=row["target"].clone();if t["kind"]=="STITCH_ROW"{t.as_object_mut().unwrap().remove("epsilon_m");t.as_object_mut().unwrap().remove("rest_length_m");}
            let request=serde_json::to_vec(&json!({"kind":"ppf.restoring_curves.mass_request","version":1,"scene_uuid":base.context.scene_uuid,"basis_sha256":base.context.scene_inputs_sha256,"targets":[{"target":t,"needs_mass":row["mass"].get("refusal").is_none()}]})).unwrap();
            let response=query(&scene,&data,&base,&request,crate::compiled_restoring_identity()).unwrap();queries.push(json!({"request_sha256":hash(&request),"request":serde_json::from_slice::<Value>(&request).unwrap(),"response":response}));}
        let after=restoring_curve::load_base_context(std::path::Path::new(&session)).unwrap();
        assert_eq!(base.bytes,after.bytes,"sealed base input bytes changed");
        assert_eq!(base.control_bytes,after.control_bytes,"sealed control bytes changed");
        assert_eq!(context_before,std::fs::read(&context_path).unwrap(),"sealed base context bytes changed");
        std::fs::write(out.join("dataset.json"),serde_json::to_vec_pretty(&raw).unwrap()).unwrap();std::fs::write(out.join("queries.json"),serde_json::to_vec_pretty(&queries).unwrap()).unwrap();
        println!("ACTUAL_BUILDER_DATASET_QA targets={} no_device_no_solver_run output={}",targets.len(),out.display());
    }
}
