//! Version/feature gate. Call before install. No unconditional Ready default.
use crate::{joint_angular_hard_wire::{AngularHardHingeWireV1,HARD_FEATURE},joint_d6_hard_wire::{D6Wire,FEATURE}};
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum RuntimeSupport { Unwired, FrozenCoupledInverseAndCandidateGates }
#[derive(Clone,Debug,Default)]
pub struct HardPresence<'a> {
    pub hinge:Option<&'a AngularHardHingeWireV1>,pub d6:Option<&'a D6Wire>,
    pub control:bool,pub soft_limits:bool,pub drives:bool,
    pub world_endpoint:bool,pub isotropic_terms:bool,pub hard_soft_conflict:bool,
}
pub fn validate_contract(version:u32, joints:&[HardPresence<'_>], features:&[String],
    root_count:usize, runtime:RuntimeSupport)->Result<(),&'static str> {
    let hinges=joints.iter().filter(|j|j.hinge.is_some()).count();
    let d6s=joints.iter().filter(|j|j.d6.is_some()).count();
    if hinges==0&&d6s==0 {
        if features.iter().any(|f|f==HARD_FEATURE||f==FEATURE) { return Err("JOINT_G2_REQUIRED_FEATURES"); }
        return Ok(()); // caller's V2/V3/V4 gate remains authoritative
    }
    if (hinges>0&&(version!=5||hinges>2||d6s!=0||root_count!=0)) ||
       (d6s>0&&(version!=7||d6s!=1||hinges!=0||root_count>1)) {
        return Err("G2_HARD_UNSUPPORTED_DOMAIN");
    }
    for j in joints {
        if let Some(h)=j.hinge { h.validate().map_err(|_|"JOINT_ANGULAR_EXTENSION_INVALID")?;
            if !j.control { return Err("JOINT_ANGULAR_EXTENSION_REQUIRES_CONTROL"); }
        }
        if let Some(d)=j.d6 { d.validate().map_err(|_|"G2_D6_INVALID_AXIS")?; }
        if j.hinge.is_some()||j.d6.is_some() {
            if j.world_endpoint||j.isotropic_terms { return Err("G2_HARD_UNSUPPORTED_DOMAIN"); }
            if j.hard_soft_conflict || (j.hinge.is_some()&&(j.soft_limits||j.drives)) { return Err("JOINT_ANGULAR_HARD_EXTENSION_CONFLICT"); }
        }
    }
    let mut expected=Vec::new();
    if joints.iter().any(|j|j.control){expected.push("pdrd_angular_controls_v1");}
    if joints.iter().any(|j|j.drives){expected.push("pdrd_angular_drives_v1");}
    if hinges>0 {expected.push(HARD_FEATURE);}
    if joints.iter().any(|j|j.soft_limits){expected.push("pdrd_angular_soft_limits_v1");}
    if d6s>0 {expected.push(FEATURE);if root_count==1 {expected.push("pdrd_kinematic_roots_v1");}}
    expected.sort_unstable();
    if features.iter().map(String::as_str).collect::<Vec<_>>()!=expected {return Err("JOINT_G2_REQUIRED_FEATURES");}
    if runtime==RuntimeSupport::Unwired {return Err("G2_HARD_RUNTIME_UNIMPLEMENTED");}
    Ok(())
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum EdgeKind {Ordinary,Hinge,D6}
#[derive(Clone,Copy,Debug)]
pub struct Edge {pub a:u32,pub b:u32,pub enabled:bool,pub stiffness:f32,pub kind:EdgeKind}
/// Canonical old hard domains; body_free_dynamic must come from sealed PDRD
/// rows, not an authored assertion. Root body stays a FREE physical PDRD row;
/// its known motion is excluded separately by the solver's mask.
pub fn validate_topology(version:u32,bodies_free_dynamic:&[bool],edges:&[Edge],known_root:Option<u32>)->Result<u32,&'static str> {
    use std::collections::HashSet;
    const ERROR:&str="G2_HARD_UNSUPPORTED_DOMAIN";
    let n=bodies_free_dynamic.len();
    if bodies_free_dynamic.iter().any(|&free|!free)||n<2 {return Err(ERROR);}
    let mut pairs=HashSet::new();
    for e in edges {if e.a as usize>=n||e.b as usize>=n||e.a==e.b||!pairs.insert((e.a.min(e.b),e.a.max(e.b))) {return Err(ERROR);}}
    let active:Vec<&Edge>=if version==5 {
        if known_root.is_some(){return Err(ERROR);}
        edges.iter().filter(|e|e.kind==EdgeKind::Hinge&&e.enabled).collect()
    }else if version==7 {
        if n>128||edges.is_empty()||edges.len()>127||edges.iter().filter(|e|e.kind==EdgeKind::D6).count()!=1||edges.iter().any(|e|e.kind==EdgeKind::Hinge){return Err(ERROR);}
        for e in edges {if !e.stiffness.is_finite()||e.stiffness<0.||(e.kind==EdgeKind::Ordinary&&(!e.enabled||e.stiffness<=0.)){return Err(ERROR);}}
        edges.iter().collect()
    }else{return Err(ERROR);};
    if version==5 && active.is_empty(){return Ok(u32::MAX);}
    if version==5&&(active.len()>2||n!=active.len()+1){return Err(ERROR);}
    let mut children=vec![Vec::new();n];let mut indegree=vec![0;n];let mut touched=HashSet::new();
    for e in active.iter() {let a=e.a as usize;let b=e.b as usize;children[a].push(b);indegree[b]+=1;touched.insert(a);touched.insert(b);
        if indegree[b]>1||(version==5&&children[a].len()>1){return Err(ERROR);}}
    if touched.len()!=active.len()+1||n-touched.len()>usize::from(version==7){return Err(ERROR);}
    let sources:Vec<usize>=touched.iter().copied().filter(|&i|indegree[i]==0).collect();if sources.len()!=1{return Err(ERROR);}let source=sources[0];
    if known_root.is_some_and(|r|r as usize!=source){return Err(ERROR);}
    let mut seen=HashSet::new();let mut stack=vec![source];while let Some(a)=stack.pop(){if !seen.insert(a){return Err(ERROR);}stack.extend(&children[a]);}
    if seen.len()!=touched.len(){return Err(ERROR);}Ok(source as u32)
}
