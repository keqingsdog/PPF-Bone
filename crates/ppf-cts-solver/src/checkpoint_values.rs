//! CPU checkpoint values and the production acceptance gates shared with tests.
#[derive(Clone,Debug,PartialEq,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Controller {
    pub(crate) authored: f32,
    pub(crate) ceiling: f32,
    pub(crate) accepted: u64,
}

impl Controller {
    pub fn validate_checkpoint(&self,authored:f32)->Result<(),String>{
        if !authored.is_finite() || authored<=0. || self.authored.to_bits()!=authored.to_bits() ||
            !self.ceiling.is_finite() || self.ceiling<=0. || self.ceiling>self.authored {
            return Err("CHECKPOINT_ADAPTIVE_INVALID: authored/sticky ceiling mismatch".into());
        }
        Ok(())
    }
    pub fn new(builder_clamped_authored: f32) -> Self {
        Self { authored: builder_clamped_authored, ceiling: builder_clamped_authored, accepted: 0 }
    }
}

/// Accepted substep history; epoch is incremented only after accepted-time transition.
/// Anchor d0, stitch q0; initialized is 0/1. Never mutated by apply kernels.
#[repr(C)]
#[derive(Clone,Copy,Debug,Default,PartialEq,serde::Serialize,serde::Deserialize)]
pub struct CurveHistory { pub d0:[f32;3], pub q0:f32, pub epoch:u64, pub initialized:u32, pub reserved:u32 }

pub fn validate_running_rotation(values:&[f32],bodies:usize)->Result<(),String>{
    let prefix="CHECKPOINT_ROTATION_INVALID: proper finite running rotation/body count required";
    if bodies.checked_mul(9)!=Some(values.len()) {
        return Err(format!("{prefix}; reason=length bodies={bodies} expected={:?} actual={}",bodies.checked_mul(9),values.len()));
    }
    for (body,m) in values.chunks_exact(9).enumerate(){
        if let Some(element)=m.iter().position(|v|!v.is_finite()) {
            return Err(format!("{prefix}; reason=nonfinite body={body} element={element} matrix={m:?}"));
        }
        let mut squared=0.0f32;
        for i in 0..3{for j in 0..3{let dot=(0..3).map(|r|m[3*i+r]*m[3*j+r]).sum::<f32>();let e=dot-if i==j{1.}else{0.};squared+=e*e;}}
        let det=m[0]*(m[4]*m[8]-m[7]*m[5])-m[3]*(m[1]*m[8]-m[7]*m[2])+m[6]*(m[1]*m[5]-m[4]*m[2]);
        let orth=squared.sqrt(); let det_error=(det-1.).abs();
        if !squared.is_finite() || !orth.is_finite() || !det.is_finite() || !det_error.is_finite() || det<=0. || orth>2e-4||det_error>2e-4 {
            return Err(format!("{prefix}; reason=improper body={body} matrix={m:?} orth_frob={orth:.9e} det={det:.9e} det_error={det_error:.9e}"));
        }
    }
    Ok(())
}

/// Validate physical restoring history without device types. The index slice
/// contains binding history indices; u32::MAX is the existing no-history sentinel.
pub fn validate_curve_history(history:&[CurveHistory],dt:f32,accepted_time_bits:Option<u64>,epoch:u64,used_history_indices:&[u32],time:f64)->Result<(),String> {
    let invalid=|detail:&str|format!("CHECKPOINT_RESTORING_HISTORY_INVALID: {detail}");
    let count=used_history_indices.iter().copied().filter(|i|*i!=u32::MAX).map(|i|i as usize+1).max().unwrap_or(0).max(1);
    if !time.is_finite() || time<0. || history.len()!=count || !dt.is_finite() || dt<0.
        || (epoch==0)!=accepted_time_bits.is_none() || (epoch>0 && dt<=0.) {return Err(invalid("layout/clock/epoch"));}
    if let Some(bits)=accepted_time_bits {
        let accepted=f64::from_bits(bits);
        if !accepted.is_finite() || accepted<0. || accepted>time || (accepted==0. && bits!=0) {return Err(invalid("accepted capture clock"));}
    }
    for row in history {
        if row.reserved!=0 || row.initialized>1 || row.epoch>epoch || !row.q0.is_finite() || !row.d0.iter().all(|v|v.is_finite())
            || (row.initialized==0 && row.epoch!=0) || (row.initialized==1 && row.epoch==0) {return Err(invalid("row scalar/init/epoch"));}
    }
    if epoch>0 {
        for index in used_history_indices.iter().copied().filter(|i|*i!=u32::MAX) {
            let row=&history[index as usize];
            if row.initialized!=1 || row.epoch!=epoch {return Err(invalid("required accepted epoch"));}
        }
    }
    Ok(())
}

/// The actual versioned sidecar reader calls this before interpreting history.
/// State's frame/time/prev_dt binding remains a separate caller contract.
pub fn validate_history_header(kind:&str,version:u32,frame:i32,expected_frame:i32,time:f64,previous_dt:f32)->Result<(),String> {
    if kind!="ppf.bone_checkpoint_history" || version!=1 || frame<0 || expected_frame<0 || frame!=expected_frame
        || !time.is_finite() || time<0. || !previous_dt.is_finite() || previous_dt<=0. {
        return Err("CHECKPOINT_HISTORY_INVALID: version/frame/clock".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn running_rotation_bits_and_properness(){
        let rotation=vec![0.0f32,1.,0.,-1.,0.,0.,0.,0.,1.];validate_running_rotation(&rotation,1).unwrap();
        let restored:Vec<f32>=bincode::deserialize(&bincode::serialize(&rotation).unwrap()).unwrap();
        assert_eq!(rotation.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),restored.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        assert!(validate_running_rotation(&rotation,2).is_err());let mut reflected=rotation.clone();reflected[8]=-1.;assert!(validate_running_rotation(&reflected,1).is_err());
        reflected[8]=f32::NAN;assert!(validate_running_rotation(&reflected,1).is_err());validate_running_rotation(&[],0).unwrap();
    }
    #[test]fn running_rotation_diagnostics_preserve_gate(){
        let identity=[1.,0.,0.,0.,1.,0.,0.,0.,1.];
        assert!(validate_running_rotation(&identity,1).is_ok());
        let length=validate_running_rotation(&identity,2).unwrap_err();
        assert!(length.contains("reason=length") && length.contains("bodies=2") && length.contains("actual=9"));
        let mut bad=identity;bad[5]=f32::NAN;
        let nonfinite=validate_running_rotation(&bad,1).unwrap_err();
        assert!(nonfinite.contains("reason=nonfinite") && nonfinite.contains("body=0") && nonfinite.contains("element=5"));
        let mut rows=identity.to_vec(); rows.extend(identity); rows[9+8]=1.001;
        let improper=validate_running_rotation(&rows,2).unwrap_err();
        assert!(improper.contains("reason=improper") && improper.contains("body=1") && improper.contains("orth_frob=") && improper.contains("det_error="));
        let mut below=identity;below[8]=1.00001;
        assert!(validate_running_rotation(&below,1).is_ok());
        assert!(validate_running_rotation(&[f32::MAX;9],1).is_err());
        assert!(validate_running_rotation(&[f32::MAX,-f32::MAX,f32::MAX,-f32::MAX,f32::MAX,-f32::MAX,f32::MAX,-f32::MAX,f32::MAX],1).is_err());
    }
    fn row()->CurveHistory {CurveHistory{d0:[-0.,f32::from_bits(0x3f123456),-0.75],q0:f32::from_bits(0x3e987654),epoch:7,initialized:1,reserved:0}}
    fn same_bits(a:CurveHistory,b:CurveHistory) {
        assert_eq!(a.d0.map(f32::to_bits),b.d0.map(f32::to_bits));assert_eq!(a.q0.to_bits(),b.q0.to_bits());
        assert_eq!((a.epoch,a.initialized,a.reserved),(b.epoch,b.initialized,b.reserved));
    }
    #[test]fn actual_curve_history_serde_bincode_roundtrip_preserves_bits_and_layout(){
        let original=row();let bytes=bincode::serialize(&original).unwrap();let decoded:CurveHistory=bincode::deserialize(&bytes).unwrap();same_bits(original,decoded);
        let json=serde_json::to_vec(&original).unwrap();let decoded:CurveHistory=serde_json::from_slice(&json).unwrap();same_bits(original,decoded);
        assert_eq!(std::mem::size_of::<CurveHistory>(),32);assert_eq!(std::mem::align_of::<CurveHistory>(),8);
        assert_eq!(std::mem::offset_of!(CurveHistory,epoch),16);assert_eq!(std::mem::offset_of!(CurveHistory,initialized),24);
        validate_curve_history(&[decoded],0.01,Some(0.9f64.to_bits()),7,&[0],1.).unwrap();
    }
    #[test]fn actual_sticky_controller_roundtrip_preserves_ceiling_and_accepted_count(){
        let mut original=Controller::new(0.01);original.ceiling=f32::from_bits(0x3b03126f);original.accepted=33;
        original.validate_checkpoint(0.01).unwrap();
        let bytes=bincode::serialize(&original).unwrap();let decoded:Controller=bincode::deserialize(&bytes).unwrap();assert_eq!(bytes,bincode::serialize(&decoded).unwrap());
        assert_eq!(original.authored.to_bits(),decoded.authored.to_bits());assert_eq!(original.ceiling.to_bits(),decoded.ceiling.to_bits());assert_eq!(decoded.accepted,33);
        let json=serde_json::to_vec(&original).unwrap();let decoded:Controller=serde_json::from_slice(&json).unwrap();assert_eq!(decoded,original);
        for ceiling in [0.,f32::NAN,f32::INFINITY,0.02] {let mut bad=original.clone();bad.ceiling=ceiling;assert!(bad.validate_checkpoint(0.01).is_err());}
        assert!(original.validate_checkpoint(0.02).is_err());
    }
    #[test]fn production_curve_gate_refuses_nan_clock_epoch_init_and_missing_row(){
        let good=row();let validate=|h:&[CurveHistory],dt,time_bits,epoch,indices:&[u32],time|validate_curve_history(h,dt,time_bits,epoch,indices,time);
        for field in 0..4 {let mut bad=good;if field<3{bad.d0[field]=f32::NAN;}else{bad.q0=f32::INFINITY;}assert!(validate(&[bad],0.01,Some(0.9f64.to_bits()),7,&[0],1.).is_err());}
        for (dt,bits,epoch,time) in [(f32::NAN,Some(0.9f64.to_bits()),7,1.),(0.,Some(0.9f64.to_bits()),7,1.),(0.01,Some(f64::NAN.to_bits()),7,1.),(0.01,Some(1.1f64.to_bits()),7,1.),(0.01,None,7,1.),(0.01,Some(0.9f64.to_bits()),0,1.),(0.01,Some(0.9f64.to_bits()),7,f64::NAN)] {assert!(validate(&[good],dt,bits,epoch,&[0],time).is_err());}
        assert!(validate(&[good],0.01,Some(0.9f64.to_bits()),8,&[0],1.).is_err());assert!(validate(&[],0.01,Some(0.9f64.to_bits()),7,&[0],1.).is_err());
        assert!(validate(&[good],0.01,Some(0.9f64.to_bits()),7,&[1],1.).is_err());let mut bad=good;bad.initialized=0;assert!(validate(&[bad],0.01,Some(0.9f64.to_bits()),7,&[0],1.).is_err());
        validate(&[CurveHistory::default()],0.,None,0,&[u32::MAX],0.).unwrap();
    }
    #[test]fn production_header_gate_rejects_wrong_version_frame_kind_and_clock(){
        let kind="ppf.bone_checkpoint_history";validate_history_header(kind,1,30,30,1.01,0.01).unwrap();
        assert!(validate_history_header(kind,2,30,30,1.01,0.01).is_err());assert!(validate_history_header(kind,1,29,30,1.01,0.01).is_err());
        assert!(validate_history_header("other",1,30,30,1.01,0.01).is_err());assert!(validate_history_header(kind,1,-1,-1,1.,0.01).is_err());
        for time in [-1.,f64::NAN,f64::INFINITY] {assert!(validate_history_header(kind,1,30,30,time,0.01).is_err());}
        for dt in [0.,-1.,f32::NAN,f32::INFINITY] {assert!(validate_history_header(kind,1,30,30,1.,dt).is_err());}
    }
}
