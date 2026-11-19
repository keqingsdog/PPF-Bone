//! Isolated V1 sidecar host decoder. Not registered by this staging milestone.
use serde::Deserialize;
use sha2::{Digest,Sha256};
use crate::restoring_curve_gpu::{CurveDescriptor,CurveNode,CurveBinding};
pub const MAX_JSON:usize=32*1024*1024;
pub const ROW_PREFIX:&[u8]=b"ppf.stitch.rows/v1\\n";
type Result<T>=std::result::Result<T,String>;
fn error(code:&str,field:&str)->String{format!("{code}: {field}")}
fn finite(x:f64,field:&str)->Result<f32>{
    let y=x as f32;
    if !x.is_finite() || !y.is_finite() || (x!=0.0 && y==0.0){return Err(error("RESTORING_ARITHMETIC_FAULT",field))} Ok(y)
}
fn id(x:&str)->bool{!x.is_empty()&&x.len()<=64&&x.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'_'||b==b'-')}
fn hash(x:&str)->bool{x.len()==64&&x.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b))}
fn uuid(x:&str)->bool{x.len()==36&&x.bytes().enumerate().all(|(i,b)|if [8,13,18,23].contains(&i){b==b'-'}else{b.is_ascii_digit()||(b'a'..=b'f').contains(&b)})}
#[derive(Clone,Debug,Deserialize,PartialEq)]
#[serde(tag="kind",deny_unknown_fields)]
pub enum Target{
    #[serde(rename="JOINT")] Joint{joint_uuid:String,channel:String},
    #[serde(rename="STITCH_ROW")] Stitch{row_index:u32,rows_sha256:String,epsilon_m:f64,rest_length_m:f64},
}
impl Target{
    fn validate(&self)->Result<()> {match self{
        Self::Joint{joint_uuid,channel}=>if !uuid(joint_uuid)||!["twist_x","swing_y","swing_z","anchor"].contains(&channel.as_str()){return Err(error("RESTORING_TARGET_INVALID","joint"))},
        Self::Stitch{rows_sha256,epsilon_m,rest_length_m,..}=>{if !hash(rows_sha256)||finite(*epsilon_m,"epsilon")?<=0.||finite(*rest_length_m,"rest_length")?<=0.{return Err(error("RESTORING_TARGET_INVALID","stitch"))}},
    } Ok(())}
    fn anchor(&self)->bool{matches!(self,Self::Joint{channel,..} if channel=="anchor")}
    fn angular(&self)->bool{matches!(self,Self::Joint{channel,..} if channel!="anchor")}
}
#[derive(Clone,Debug,Deserialize)] #[serde(deny_unknown_fields)]
pub struct Table{pub table_id:String,pub domain:String,pub tail:String,pub nodes:Vec<[f64;2]>}
#[derive(Clone,Debug,Deserialize)] #[serde(deny_unknown_fields)]
pub struct Binding{pub target:Target,pub table_id:String,pub damping:f64,pub mass_receipt_id:Option<String>}
#[derive(Clone,Debug,Deserialize)] #[serde(deny_unknown_fields)]
pub struct MassReceipt{
    pub receipt_id:String,pub target:Target,pub policy:String,pub quality:String,pub basis_sha256:String,
    pub metric_eff:f64,pub metric_unit:String,pub mobility_eigenvalues:Option<[f64;3]>,
    pub zero_direction_count:Option<u32>,pub anisotropy_ratio:Option<f64>,
    pub frequency_hz:f64,pub damping_ratio:f64,pub coordinate_scale:f64,
    pub resolved_stiffness:f64,pub resolved_damping:f64,pub epsilon_m:Option<f64>,pub rest_length_m:Option<f64>,
}
#[derive(Clone,Debug,Deserialize)] #[serde(deny_unknown_fields)]
pub struct Sidecar{pub kind:String,pub version:u32,pub required_feature:String,pub scene_uuid:String,pub basis_sha256:String,pub tables:Vec<Table>,pub bindings:Vec<Binding>,pub mass_receipts:Vec<MassReceipt>}
pub struct PackedTables{pub descriptors:Vec<CurveDescriptor>,pub nodes:Vec<CurveNode>}
pub fn pack_table(table:&Table,offset:u32)->Result<(CurveDescriptor,Vec<CurveNode>)>{
    let bad=||error("RESTORING_CURVE_INVALID",&table.table_id);
    if !id(&table.table_id)||! ["SIGNED","ODD","RADIAL"].contains(&table.domain.as_str())||!["LINEAR","HOLD_EFFORT"].contains(&table.tail.as_str())||!(2..=64).contains(&table.nodes.len()){return Err(bad())}
    let mut nodes=Vec::new();
    for [x,f] in &table.nodes{nodes.push(CurveNode{x:finite(*x,"x")?,effort:finite(*f,"effort")?,..Default::default()})}
    let zeros:Vec<_>=nodes.iter().enumerate().filter(|(_,n)|n.x==0.&&n.effort==0.).map(|(i,_)|i).collect();
    if zeros.len()!=1{return Err(bad())}
    let last=nodes.len()-1;
    if table.domain=="SIGNED" {if !(nodes[0].x<0.&&nodes[last].x>0.){return Err(bad())}}
    else if zeros[0]!=0{return Err(bad())}
    let mut segments=vec![0f64;last];
    for i in 0..last{
        let dx=nodes[i+1].x as f64-nodes[i].x as f64;
        let df=nodes[i+1].effort as f64-nodes[i].effort as f64;
        if dx<=0.||df<0.{return Err(bad())}
        finite(dx,"spacing")?;finite(df,"effort difference")?;
        nodes[i].right_slope=finite(df/dx,"slope")?;
        let area=(nodes[i].effort as f64+nodes[i+1].effort as f64)*dx/2.;finite(area,"segment integral")?;
        segments[i]=area;
    }
    nodes[last].right_slope=if table.tail=="LINEAR"{nodes[last-1].right_slope}else{0.};
    let zero=zeros[0];let mut areas=vec![0f64;nodes.len()];
    for i in zero..last{areas[i+1]=areas[i]+segments[i];}
    for i in (0..zero).rev(){areas[i]=areas[i+1]-segments[i];}
    for i in 0..nodes.len(){nodes[i].area=finite(areas[i],"area")?}
    Ok((CurveDescriptor{node_offset:offset,node_count:nodes.len() as u32,domain:match table.domain.as_str(){"SIGNED"=>0,"ODD"=>1,_=>2},tail:if table.tail=="LINEAR"{0}else{1},reserved:[0;4]},nodes))
}
pub fn parse(bytes:&[u8],basis:&str)->Result<(Sidecar,PackedTables)>{
    if bytes.len()>MAX_JSON{return Err(error("RESTORING_BUDGET_EXCEEDED","JSON"))}
    // A raw Value first enforces explicit null-bearing fields as required keys.
    let raw:serde_json::Value=serde_json::from_slice(bytes).map_err(|e|error("RESTORING_SCHEMA_INVALID",&e.to_string()))?;
    for b in raw.get("bindings").and_then(|v|v.as_array()).into_iter().flatten(){if b.get("mass_receipt_id").is_none(){return Err(error("RESTORING_SCHEMA_INVALID","mass_receipt_id"))}}
    for r in raw.get("mass_receipts").and_then(|v|v.as_array()).into_iter().flatten(){for k in ["mobility_eigenvalues","zero_direction_count","anisotropy_ratio","epsilon_m","rest_length_m"]{if r.get(k).is_none(){return Err(error("RESTORING_SCHEMA_INVALID",k))}}}
    // Deserialize the original bytes as typed structs too: duplicate struct keys
    // are rejected by serde instead of disappearing in the intermediate Value.
    let doc:Sidecar=serde_json::from_slice(bytes).map_err(|e|error("RESTORING_SCHEMA_INVALID",&e.to_string()))?;
    if doc.version!=1{return Err(error("RESTORING_VERSION_UNSUPPORTED","version"))}
    if doc.kind!="ppf.restoring_curves"||doc.required_feature!="restoring_curves_v1"||!uuid(&doc.scene_uuid)||!hash(&doc.basis_sha256){return Err(error("RESTORING_SCHEMA_INVALID","envelope"))}
    if doc.basis_sha256!=basis{return Err(error("RESTORING_BASIS_MISMATCH","basis"))}
    if doc.tables.len()>4096||doc.bindings.len()>65536{return Err(error("RESTORING_BUDGET_EXCEEDED","counts"))}
    let mut packed=PackedTables{descriptors:vec![],nodes:vec![]};let mut ids=std::collections::HashSet::new();
    for t in &doc.tables{
        if !ids.insert(t.table_id.clone()){return Err(error("RESTORING_SCHEMA_INVALID","duplicate table"))}
        let (d,n)=pack_table(t,packed.nodes.len() as u32)?;packed.descriptors.push(d);packed.nodes.extend(n);
        if packed.nodes.len()>262144{return Err(error("RESTORING_BUDGET_EXCEEDED","nodes"))}
    }
    let receipt_bytes=serde_json::to_vec(&raw["mass_receipts"]).map_err(|e|error("RESTORING_SCHEMA_INVALID",&e.to_string()))?.len();
    let bytes=doc.tables.len()*32+packed.nodes.len()*16+doc.bindings.len()*48+receipt_bytes;
    if bytes>16*1024*1024{return Err(error("RESTORING_BUDGET_EXCEEDED","packed"))}
    let mut seen=Vec::new();let mut receipt_ids=std::collections::HashSet::new();
    for r in &doc.mass_receipts{if !receipt_ids.insert(&r.receipt_id){return Err(error("RESTORING_SCHEMA_INVALID","duplicate receipt"))}validate_receipt(r)?;if r.basis_sha256!=basis{return Err(error("RESTORING_BASIS_MISMATCH","receipt"))}}
    for b in &doc.bindings{
        b.target.validate()?;
        let identity=match &b.target{Target::Joint{joint_uuid,channel}=>format!("J:{joint_uuid}:{channel}"),Target::Stitch{row_index,..}=>format!("S:{row_index}")};
        if seen.contains(&identity){return Err(error("RESTORING_CONFLICT","target"))}seen.push(identity);
        if finite(b.damping,"damping")?<0.{return Err(error("RESTORING_CURVE_INVALID","damping"))}
        let t=doc.tables.iter().find(|t|t.table_id==b.table_id).ok_or_else(||error("RESTORING_TARGET_INVALID","table reference"))?;
        if b.target.anchor()!=(t.domain=="RADIAL"){return Err(error("RESTORING_TARGET_INVALID","domain"))}
        if b.target.angular()&&t.nodes.iter().any(|n|n[0].abs() as f32>=std::f32::consts::PI-0.001){return Err(error("RESTORING_CURVE_INVALID","angular range"))}
        if let Some(rid)=&b.mass_receipt_id{
            let r=doc.mass_receipts.iter().find(|r|&r.receipt_id==rid).ok_or_else(||error("RESTORING_MASS_INVALID","receipt reference"))?;
            if r.target!=b.target||r.basis_sha256!=basis||r.resolved_damping!=b.damping{return Err(error("RESTORING_MASS_INVALID","receipt identity"))}
            validate_receipt(r)?;
        }
    }
    for t in &doc.tables{if !doc.bindings.iter().any(|b|b.table_id==t.table_id){return Err(error("RESTORING_TARGET_INVALID","unreferenced table"))}}
    for r in &doc.mass_receipts{if !doc.bindings.iter().any(|b|b.mass_receipt_id.as_ref()==Some(&r.receipt_id)){return Err(error("RESTORING_MASS_INVALID","unreferenced receipt"))}}
    Ok((doc,packed))
}
fn validate_receipt(r:&MassReceipt)->Result<()>{
    let bad=||error("RESTORING_MASS_INVALID",&r.receipt_id);
    if !id(&r.receipt_id)||r.policy!="LOCAL_REST_V1"||!["LOCAL_RIGID_ESTIMATE","ANCHOR_MEAN_COMPLIANCE_ESTIMATE","FEM_LOCAL_MASS_ESTIMATE","STITCH_NODAL_LOCAL_ESTIMATE"].contains(&r.quality.as_str()){return Err(bad())}
    r.target.validate()?;
    match &r.target{
        Target::Joint{channel,..}=>if channel=="anchor" {if !["ANCHOR_MEAN_COMPLIANCE_ESTIMATE","FEM_LOCAL_MASS_ESTIMATE"].contains(&r.quality.as_str()){return Err(bad())}}else if !["LOCAL_RIGID_ESTIMATE","FEM_LOCAL_MASS_ESTIMATE"].contains(&r.quality.as_str()){return Err(bad())},
        Target::Stitch{..}=>if r.quality!="STITCH_NODAL_LOCAL_ESTIMATE"{return Err(bad())},
    }
    for x in [r.metric_eff,r.frequency_hz,r.coordinate_scale,r.resolved_stiffness]{if finite(x,"receipt positive")?<=0.{return Err(bad())}}
    if finite(r.damping_ratio,"ratio")?<0.||finite(r.resolved_damping,"C")?<0.{return Err(bad())}
    let omega=2.*std::f64::consts::PI*r.frequency_hz;
    if finite(r.metric_eff*omega*omega,"K")?!=r.resolved_stiffness as f32||finite(2.*r.damping_ratio*r.metric_eff*omega,"C")?!=r.resolved_damping as f32{return Err(bad())}
    if r.metric_unit!=if r.target.angular(){"kg_m2"}else{"kg"}{return Err(bad())}
    if r.target.anchor(){
        let eig=r.mobility_eigenvalues.ok_or_else(bad)?; let mut positive=Vec::new();let mut zeros=0;
        for x in eig{if !x.is_finite()||x<0.{return Err(bad())}if x==0.{zeros+=1}else{positive.push(x)}}
        if positive.is_empty()||r.zero_direction_count!=Some(zeros){return Err(bad())}
        let ratio=positive.iter().copied().fold(0.,f64::max)/positive.iter().copied().fold(f64::INFINITY,f64::min);
        if r.anisotropy_ratio!=Some(ratio)||(1./(eig.iter().sum::<f64>()/3.)-r.metric_eff).abs()>r.metric_eff*1e-6{return Err(bad())}
    }else if r.mobility_eigenvalues.is_some()||r.zero_direction_count.is_some()||r.anisotropy_ratio.is_some(){return Err(bad())}
    match &r.target{Target::Stitch{epsilon_m,rest_length_m,..}=>if r.epsilon_m!=Some(*epsilon_m)||r.rest_length_m!=Some(*rest_length_m){return Err(bad())},_=>if r.epsilon_m.is_some()||r.rest_length_m.is_some(){return Err(bad())}}
    Ok(())
}
pub fn rows_hash(ind:&[u8],weights:&[u8],count:u64)->Result<String>{
    if ind.len()%48!=0||weights.len()%24!=0||ind.len()/48!=weights.len()/24||count!=ind.len() as u64/48{return Err(error("RESTORING_TARGET_INVALID","stitch lengths/count"))}
    for row in weights.chunks_exact(24){
        let w:Vec<_>=row.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        if w.iter().any(|x|!x.is_finite()||*x<0.)||(w[0..3].iter().sum::<f32>()-1.).abs()>1e-5||(w[3..6].iter().sum::<f32>()-1.).abs()>1e-5{return Err(error("RESTORING_TARGET_INVALID","stitch barycentric weight"))}
    }
    let mut h=Sha256::new();h.update(ROW_PREFIX);h.update(count.to_le_bytes());h.update(ind);h.update(weights);Ok(format!("{:x}",h.finalize()))
}
/// Native-local nodal estimate with shared vertices merged first. fixed inverse
/// mass is exactly zero. Coefficients must come from actual rest Jacobian/rows.
pub fn nodal_metric(coefficients:&[(usize,f64)],masses:&[f64],fixed:&[bool])->Result<f64>{
    if masses.len()!=fixed.len(){return Err(error("RESTORING_MASS_INVALID","mass/lock count"))}
    let mut merged=std::collections::BTreeMap::<usize,f64>::new();
    for (v,c) in coefficients{if *v>=masses.len()||!c.is_finite(){return Err(error("RESTORING_MASS_INVALID","nodal coefficient"))}*merged.entry(*v).or_default()+=c;}
    let mut mu=0.;for (v,c) in merged{if fixed[v]||c==0.{continue}let m=masses[v];if !m.is_finite()||m<=0.{return Err(error("RESTORING_MASS_INVALID","nodal mass"))}mu+=c*c/m;}
    if !mu.is_finite()||mu<=0.{return Err(error("RESTORING_MASS_INVALID","zero mobility"))}let metric=1./mu;finite(metric,"metric")?;Ok(metric)
}
/// Pack separate records only after identity, target conflict, row hash, and mass
/// validation. target_index is actual joint/FEM/row index, never a GUI ordinal.
/// resolver returns (kind,index); rigid/FEM angular 0/1, anchor2/3, stitch4.
pub fn pack_bindings<F>(doc:&Sidecar,mut resolver:F)->Result<Vec<CurveBinding>>where F:FnMut(&Target)->Result<(u32,u32)>{
    let mut out=Vec::with_capacity(doc.bindings.len());let mut history=0u32;let mut aliases=std::collections::HashSet::new();
    for b in &doc.bindings{
        let(kind,index)=resolver(&b.target)?;
        let channel=match &b.target{Target::Joint{channel,..}=>match channel.as_str(){"twist_x"=>0,"swing_y"=>1,"swing_z"=>2,"anchor"=>3,_=>return Err(error("RESTORING_TARGET_INVALID","channel"))},_=>3};
        if kind>4||b.target.angular()!=(kind<=1)||b.target.anchor()!=(kind==2||kind==3)||matches!(b.target,Target::Stitch{..})!=(kind==4){return Err(error("RESTORING_TARGET_INVALID","native kind"))}
        if !aliases.insert((kind,index,channel)){return Err(error("RESTORING_CONFLICT","native binding alias"))}
        let table_index=doc.tables.iter().position(|t|t.table_id==b.table_id).ok_or_else(||error("RESTORING_TARGET_INVALID","table"))? as u32;
        let receipt_index=match &b.mass_receipt_id{None=>u32::MAX,Some(r)=>doc.mass_receipts.iter().position(|v|&v.receipt_id==r).ok_or_else(||error("RESTORING_MASS_INVALID","receipt"))? as u32};
        let history_index=if kind>=2{let h=history;history=history.checked_add(1).ok_or_else(||error("RESTORING_BUDGET_EXCEEDED","history"))?;h}else{u32::MAX};
        let(epsilon,rest)=match &b.target{Target::Stitch{epsilon_m,rest_length_m,..}=>(finite(*epsilon_m,"epsilon")?,finite(*rest_length_m,"rest")?),_=>(0.,0.)};
        out.push(CurveBinding{kind,target_index:index,channel,table_index,history_index,receipt_index,damping:finite(b.damping,"damping")?,epsilon_m:epsilon,rest_length_m:rest,reserved:[0;3]});
    }Ok(out)
}
/// Compare row inventory before upload. Same complete digest on every row.
pub fn verify_stitch_rows(doc:&Sidecar,ind:&[u8],weights:&[u8],count:u64)->Result<()>{
    let hash=rows_hash(ind,weights,count)?;
    for b in &doc.bindings{if let Target::Stitch{row_index,rows_sha256,..}=&b.target{if u64::from(*row_index)>=count||rows_sha256!=&hash{return Err(error("RESTORING_TARGET_INVALID","row hash/index"))}}}Ok(())
}
/// Independent native recomputation is mandatory before any upload. This callback
/// receives sealed solver input authority, not a receipt claimed by the addon.
pub struct MassEvidence{
    pub metric_eff:f64,pub quality:String,pub mobility_eigenvalues:Option<[f64;3]>,
    pub zero_direction_count:Option<u32>,pub anisotropy_ratio:Option<f64>,
}
fn close(a:f64,b:f64)->bool{a.is_finite()&&b.is_finite()&&(a-b).abs()<=a.abs().max(b.abs())*1e-6}
pub fn verify_mass<F>(doc:&Sidecar,mut recompute:F)->Result<()>where F:FnMut(&Target)->Result<MassEvidence>{
    for r in &doc.mass_receipts{
        let actual=recompute(&r.target)?;
        let eig_equal=match(actual.mobility_eigenvalues,r.mobility_eigenvalues){(None,None)=>true,(Some(a),Some(b))=>a.iter().zip(b).all(|(x,y)|close(*x,y)),_=>false};
        let ratio_equal=match(actual.anisotropy_ratio,r.anisotropy_ratio){(None,None)=>true,(Some(a),Some(b))=>close(a,b),_=>false};
        if actual.metric_eff<=0.||!close(actual.metric_eff,r.metric_eff)||actual.quality!=r.quality||!eig_equal||actual.zero_direction_count!=r.zero_direction_count||!ratio_equal{return Err(error("RESTORING_MASS_INVALID",&r.receipt_id))}
    }Ok(())
}
/// Entry gate contract: called globally before ordinary joint preparation. Capability
/// must be checked by C's actual opened Device, before invoking this decoder.
pub fn prepare_fresh(path:&std::path::Path,require:bool,load:i32)->Result<Option<PreparedCurves>>{
    let file=path.join("bin/restoring_curves_v1.json");
    match std::fs::symlink_metadata(&file){
        Err(e) if e.kind()==std::io::ErrorKind::NotFound=>{
            if require{return Err(error("RESTORING_CAPABILITY_UNAVAILABLE","required sidecar absent"))}
            // Marker alone is a persistent enabled feature requirement too.
            // Old contexts with no marker retain their own validation path.
            if path.join("bin/pdrd_scene_context.cbor").exists(){
                let raw=read_safe(path,"bin/pdrd_scene_context.cbor")?;
                if let Ok(ciborium::Value::Map(envelope))=ciborium::de::from_reader::<ciborium::Value,_>(raw.as_slice()){
                    for(k,v)in envelope{if k.as_text()==Some("payload"){if let ciborium::Value::Map(payload)=v{if payload.iter().any(|(key,_)|key.as_text()==Some("restoring_curves_v1")){return Err(error("RESTORING_BASIS_MISMATCH","sealed restoring sidecar absent"))}}}}
                }
            }
            return Ok(None)
        },
        Err(e)=>return Err(error("RESTORING_BASIS_MISMATCH",&e.to_string())),
        Ok(_)=>if unsafe_meta(&file)?{return Err(error("RESTORING_BASIS_MISMATCH","unsafe extension input"))},
    }
    if load!=0{return Err(error("RESTORING_CONFLICT","checkpoint history unsupported"))}
    let snapshot=load_base_context(path)?;
    let seal=snapshot.context.restoring_curves_v1.as_ref().ok_or_else(||error("RESTORING_BASIS_MISMATCH","unsealed restoring sidecar"))?;
    let bytes=read_safe(path,"bin/restoring_curves_v1.json")?;
    let sidecar_hash=format!("{:x}",Sha256::digest(&bytes));
    let combined=format!("{:x}",Sha256::digest(format!("{}{}",snapshot.context.scene_inputs_sha256,sidecar_hash).as_bytes()));
    if seal.required_feature!="restoring_curves_v1"||seal.sidecar_byte_length!=bytes.len()||seal.sidecar_sha256!=sidecar_hash||seal.combined_inputs_sha256!=combined{return Err(error("RESTORING_BASIS_MISMATCH","extension seal"))}
    let(sidecar,packed)=parse(&bytes,&snapshot.context.scene_inputs_sha256)?;
    if sidecar.scene_uuid!=snapshot.context.scene_uuid{return Err(error("RESTORING_BASIS_MISMATCH","scene UUID"))}
    Ok(Some(PreparedCurves{sidecar,packed,snapshot,sidecar_bytes:bytes,sidecar_sha256:sidecar_hash,combined_inputs_sha256:combined}))
}
#[derive(Deserialize)] #[serde(deny_unknown_fields)]
pub struct ExtensionSeal{pub required_feature:String,pub sidecar_sha256:String,pub sidecar_byte_length:usize,pub combined_inputs_sha256:String}
#[derive(Deserialize)] #[serde(deny_unknown_fields)]
pub struct ContextBody{pub body_uuid:String,pub runtime_body_index:u32,pub vertex_count:u32}
#[derive(Deserialize)] #[serde(deny_unknown_fields)]
pub struct ContextInput{pub relative_path:String,pub byte_length:usize,pub sha256:String}
#[derive(Deserialize)] #[serde(deny_unknown_fields)]
pub struct BaseContext{
    pub context_version:u32,pub scene_uuid:String,pub builder_contract_version:u32,pub complete:bool,
    pub bodies:Vec<ContextBody>,pub inputs:Vec<ContextInput>,pub optional_absent:Vec<String>,pub scene_inputs_sha256:String,
    #[serde(default)] pub restoring_curves_v1:Option<ExtensionSeal>,
}
#[derive(Deserialize)] #[serde(deny_unknown_fields)]
struct ContextEnvelope{version:u32,kind:String,payload:BaseContext}
pub struct BaseSnapshot{pub context:BaseContext,context_bytes:Vec<u8>,pub bytes:std::collections::BTreeMap<String,Vec<u8>>,pub control_bytes:std::collections::BTreeMap<String,Vec<u8>>}
pub struct PreparedCurves{
    pub sidecar:Sidecar,pub packed:PackedTables,pub snapshot:BaseSnapshot,
    sidecar_bytes:Vec<u8>,pub sidecar_sha256:String,pub combined_inputs_sha256:String,
}
fn unsafe_meta(path:&std::path::Path)->Result<bool>{
    let meta=std::fs::symlink_metadata(path).map_err(|e|error("RESTORING_BASIS_MISMATCH",&e.to_string()))?;
    if meta.file_type().is_symlink(){return Ok(true)}
    #[cfg(windows)]{use std::os::windows::fs::MetadataExt;if meta.file_attributes()&0x400!=0{return Ok(true)}}
    Ok(false)
}
fn read_safe(root:&std::path::Path,relative:&str)->Result<Vec<u8>>{
    use std::path::{Path,Component};
    if relative.is_empty()||relative.contains('\\')||relative.contains(':')||Path::new(relative).components().any(|c|!matches!(c,Component::Normal(_))){return Err(error("RESTORING_BASIS_MISMATCH","unsafe input path"))}
    let absolute=if root.is_absolute(){root.to_path_buf()}else{std::env::current_dir().map_err(|e|error("RESTORING_BASIS_MISMATCH",&e.to_string()))?.join(root)};
    for p in absolute.ancestors(){if unsafe_meta(p)?{return Err(error("RESTORING_BASIS_MISMATCH","unsafe scene root"))}}
    let mut file=absolute;
    for part in Path::new(relative).components(){file.push(part);if unsafe_meta(&file)?{return Err(error("RESTORING_BASIS_MISMATCH","unsafe input ancestor"))}}
    std::fs::read(file).map_err(|e|error("RESTORING_BASIS_MISMATCH",&e.to_string()))
}
fn base_paths(root:&std::path::Path)->Result<Vec<String>>{
    fn walk(root:&std::path::Path,dir:&std::path::Path,out:&mut Vec<String>)->Result<()>{
        for entry in std::fs::read_dir(dir).map_err(|e|error("RESTORING_BASIS_MISMATCH",&e.to_string()))?{
            let entry=entry.map_err(|e|error("RESTORING_BASIS_MISMATCH",&e.to_string()))?;
            if unsafe_meta(&entry.path())?{return Err(error("RESTORING_BASIS_MISMATCH","unsafe input tree"))}
            if entry.file_type().map_err(|e|error("RESTORING_BASIS_MISMATCH",&e.to_string()))?.is_dir(){walk(root,&entry.path(),out)?}
            else{let relative=entry.path().strip_prefix(root).unwrap().to_string_lossy().replace('\\',"/");
                if !["bin/pdrd_ball_anchor.cbor","bin/pdrd_scene_context.cbor","bin/restoring_curves_v1.json"].contains(&relative.as_str()){out.push(relative)}
            }
        }Ok(())
    }
    let mut out=Vec::new();
    for name in ["info.toml","param.toml","dyn_param.txt","map.pickle","command.bat"]{if root.join(name).exists(){out.push(name.to_owned())}}
    if root.join("bin").exists(){if unsafe_meta(&root.join("bin"))?{return Err(error("RESTORING_BASIS_MISMATCH","unsafe bin tree"))}walk(root,&root.join("bin"),&mut out)?}
    out.sort();Ok(out)
}
/// Pure-host Build query may call this before the sidecar exists. Same legacy
/// aggregate, with only the separately classified extension excluded.
pub fn load_base_context(root:&std::path::Path)->Result<BaseSnapshot>{
    let context_bytes=read_safe(root,"bin/pdrd_scene_context.cbor")?;
    let mut cursor=std::io::Cursor::new(context_bytes.as_slice());
    let envelope:ContextEnvelope=ciborium::de::from_reader(&mut cursor).map_err(|e|error("RESTORING_SCHEMA_INVALID",&e.to_string()))?;
    if cursor.position()!=context_bytes.len() as u64{return Err(error("RESTORING_SCHEMA_INVALID","trailing context CBOR"))}
    let context=envelope.payload;
    if envelope.version!=2||envelope.kind!="ppf.pdrd_scene_context"||context.context_version!=2||context.builder_contract_version!=2||!context.complete||!uuid(&context.scene_uuid)||!hash(&context.scene_inputs_sha256){return Err(error("RESTORING_BASIS_MISMATCH","base context contract"))}
    let listed:Vec<_>=context.inputs.iter().map(|v|v.relative_path.clone()).collect();
    if listed!=base_paths(root)?{return Err(error("RESTORING_BASIS_MISMATCH","base inventory"))}
    let mut sorted=listed.clone();sorted.sort();sorted.dedup();if sorted!=listed{return Err(error("RESTORING_BASIS_MISMATCH","duplicate/unsorted inventory"))}
    let mut bodies:Vec<_>=context.bodies.iter().collect();bodies.sort_by(|a,b|a.body_uuid.cmp(&b.body_uuid));
    let mut digest=format!("ppf.pdrd_scene_context/v2\n{}\n",context.scene_uuid);let mut seen=std::collections::HashSet::new();let mut indexes=std::collections::HashSet::new();
    for b in bodies{if !uuid(&b.body_uuid)||!seen.insert(&b.body_uuid)||!indexes.insert(b.runtime_body_index){return Err(error("RESTORING_BASIS_MISMATCH","base body identities"))}digest.push_str(&format!("{}\t{}\t{}\n",b.body_uuid,b.runtime_body_index,b.vertex_count))}
    let mut bytes=std::collections::BTreeMap::new();
    for i in &context.inputs{let data=read_safe(root,&i.relative_path)?;if data.len()!=i.byte_length||format!("{:x}",Sha256::digest(&data))!=i.sha256{return Err(error("RESTORING_BASIS_MISMATCH","base input hash"))}digest.push_str(&format!("{}\t{}\t{}\n",i.relative_path,i.byte_length,i.sha256));bytes.insert(i.relative_path.clone(),data);}
    let mut absent=context.optional_absent.clone();absent.sort();absent.dedup();if absent!=context.optional_absent{return Err(error("RESTORING_BASIS_MISMATCH","absence order"))}
    for p in &absent{if !["dyn_param.txt","bin/rest_vert.bin","bin/rest_vert_anim.bin","bin/pdrd_root_motion_v1.cbor"].contains(&p.as_str())||std::fs::symlink_metadata(root.join(p)).is_ok(){return Err(error("RESTORING_BASIS_MISMATCH","optional absence"))}digest.push_str(&format!("optional_absent\t{p}\n"));}
    if format!("{:x}",Sha256::digest(digest.as_bytes()))!=context.scene_inputs_sha256{return Err(error("RESTORING_BASIS_MISMATCH","base digest"))}
    // Ordinary joint manifest stays outside its own legacy aggregate but is
    // still consumed control data. Freeze it separately, including absence.
    let mut control_bytes=std::collections::BTreeMap::new();
    if root.join("bin/pdrd_ball_anchor.cbor").exists(){control_bytes.insert("bin/pdrd_ball_anchor.cbor".to_owned(),read_safe(root,"bin/pdrd_ball_anchor.cbor")?);}
    Ok(BaseSnapshot{context,context_bytes,bytes,control_bytes})
}
impl PreparedCurves{
    /// Invoke before post-DataSet mass recomputation and again before upload.
    pub fn verify_unchanged(&self,root:&std::path::Path)->Result<()>{
        let now=load_base_context(root)?;
        if now.context_bytes!=self.snapshot.context_bytes||now.bytes!=self.snapshot.bytes||now.control_bytes!=self.snapshot.control_bytes||read_safe(root,"bin/restoring_curves_v1.json")?!=self.sidecar_bytes{return Err(error("RESTORING_BASIS_MISMATCH","input changed after preparation"))}Ok(())
    }
}

#[cfg(test)]
mod receipt_roundtrip_tests{
    use super::*;
    // Keep the exact capsule receipt numbers local to the test: historical
    // coordination artifacts are not part of a portable source checkout.
    const REAL_CAPSULE_RECEIPT:&[u8]=br#"{
        "kind":"ppf.restoring_curves","version":1,"required_feature":"restoring_curves_v1",
        "scene_uuid":"11111111-1111-4111-8111-111111111111",
        "basis_sha256":"0000000000000000000000000000000000000000000000000000000000000000",
        "tables":[{"table_id":"capsule_anchor","domain":"RADIAL","tail":"LINEAR","nodes":[[0,0],[1,41.37961196899414]]}],
        "bindings":[{"target":{"kind":"JOINT","joint_uuid":"22222222-2222-4222-8222-222222222222","channel":"anchor"},"table_id":"capsule_anchor","damping":0.06146718189120293,"mass_receipt_id":"capsule_mass"}],
        "mass_receipts":[{
            "receipt_id":"capsule_mass","target":{"kind":"JOINT","joint_uuid":"22222222-2222-4222-8222-222222222222","channel":"anchor"},
            "policy":"LOCAL_REST_V1","quality":"ANCHOR_MEAN_COMPLIANCE_ESTIMATE",
            "basis_sha256":"0000000000000000000000000000000000000000000000000000000000000000",
            "metric_eff":0.000046584789894526544,"metric_unit":"kg",
            "mobility_eigenvalues":[21376.459083428268,21511.121435207235,21511.121435207235],
            "zero_direction_count":0,"anisotropy_ratio":1.0062995630498675,
            "frequency_hz":150,"damping_ratio":0.7,"coordinate_scale":1,
            "resolved_stiffness":41.37961196899414,"resolved_damping":0.06146718189120293,
            "epsilon_m":null,"rest_length_m":null
        }]
    }"#;
    fn fixture()->(Sidecar,String){let doc:Sidecar=serde_json::from_slice(REAL_CAPSULE_RECEIPT).unwrap();let basis=doc.basis_sha256.clone();(doc,basis)}
    #[test]fn capsule_roundtrip_valid_exact_bits_and_f32_k_c(){
        let (doc,basis)=fixture();let r=&doc.mass_receipts[0];
        let eig=r.mobility_eigenvalues.unwrap();
        assert_eq!(eig.map(f64::to_bits),[21376.459083428268f64,21511.121435207235,21511.121435207235].map(f64::to_bits));
        assert_eq!(r.anisotropy_ratio.unwrap().to_bits(),1.0062995630498675f64.to_bits());
        assert_eq!(r.anisotropy_ratio,Some(eig.iter().copied().fold(0.,f64::max)/eig.iter().copied().fold(f64::INFINITY,f64::min)));
        let omega=2.*std::f64::consts::PI*r.frequency_hz;
        assert_eq!(finite(r.metric_eff*omega*omega,"K").unwrap().to_bits(),41.37961196899414f32.to_bits());
        assert_eq!(finite(2.*r.damping_ratio*r.metric_eff*omega,"C").unwrap().to_bits(),0.06146718189120293f32.to_bits());
        validate_receipt(r).unwrap();parse(REAL_CAPSULE_RECEIPT,&basis).unwrap();
    }
    #[test]fn capsule_roundtrip_one_ulp_ratio_tamper_still_rejected(){
        let (mut doc,_)=fixture();let r=&mut doc.mass_receipts[0];r.anisotropy_ratio=Some(f64::from_bits(r.anisotropy_ratio.unwrap().to_bits()+1));
        assert!(validate_receipt(r).unwrap_err().starts_with("RESTORING_MASS_INVALID"));
        let text=std::str::from_utf8(REAL_CAPSULE_RECEIPT).unwrap().replace("\"anisotropy_ratio\":1.0062995630498675","\"anisotropy_ratio\":1.0062995630498677");
        let (_,basis)=fixture();assert!(parse(text.as_bytes(),&basis).is_err());
    }
    #[test]fn capsule_roundtrip_nan_duplicate_illegal_still_rejected(){
        let text=std::str::from_utf8(REAL_CAPSULE_RECEIPT).unwrap();let (_,basis)=fixture();
        for bad in [text.replace("\"anisotropy_ratio\":1.0062995630498675","\"anisotropy_ratio\":NaN"),
                    text.replace("\"anisotropy_ratio\":1.0062995630498675","\"anisotropy_ratio\":1.0062995630498675,\"anisotropy_ratio\":1.0062995630498675"),
                    text.replace("\"channel\":\"anchor\"","\"channel\":\"illegal\"")]{assert!(parse(bad.as_bytes(),&basis).is_err());}
        let (mut doc,_)=fixture();doc.mass_receipts[0].metric_eff=f64::NAN;assert!(validate_receipt(&doc.mass_receipts[0]).is_err());
    }
}
