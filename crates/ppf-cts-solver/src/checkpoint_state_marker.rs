//! Forward checkpoint provenance carried inside the existing immutable State file.
//! Ordinary State keeps its bincode layout. Plastic binds its sidecar in an envelope.
use std::borrow::Cow;
use std::io::{Read,Write};
use flate2::{Compression,GzBuilder};
use flate2::bufread::GzDecoder;

pub const MARKER: &[u8] = b"PPF_RUNTIME_HISTORY_REQUIRED_V1";
const OUR_PREFIX: &[u8] = b"PPF_RUNTIME_HISTORY_REQUIRED";
const PLASTIC_MARKER: &[u8] = b"PPF_PLASTIC_BOUND_V1";
const PLASTIC_MAGIC: &[u8] = b"PPFPLS01";
// No new hard State-size policy: the existing loader is unbounded.
// Explicit bounded callers may use decode_bounded.

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum HistoryRequirement {
    /// No forward marker: cannot establish historical sidecar requirements.
    LegacyUnmarkedAmbiguous,
    RequiredV1,
    RequiredSandV1,
}
#[derive(Debug,PartialEq,Eq)]
pub struct DecodedState { pub requirement: HistoryRequirement, pub payload: Vec<u8>, pub(crate) sand: Option<Vec<u8>>, pub(crate) plastic_binding: Option<Option<(u64,String)>> }

/// Full gzip validation includes header, payload, CRC and length; never inspect
/// a comment alone and call a corrupt/truncated state readable. Single member
/// only, as produced by State saves; reject extra bytes rather than ignore them.
pub fn decode(bytes: &[u8]) -> Result<DecodedState,String> {
    decode_bounded(bytes,u64::MAX-1)
}
pub fn decode_bounded(bytes:&[u8],limit:u64)->Result<DecodedState,String>{
    decode_member(bytes, limit, true)
}

fn decode_member(bytes:&[u8],limit:u64,allow_plastic_binding:bool)->Result<DecodedState,String>{
    let mut decoder=GzDecoder::new(bytes);
    let comment=decoder.header().and_then(|h|h.comment()).map(|b|b.to_vec());
    let bound=comment.as_deref()==Some(PLASTIC_MARKER);
    let requirement=match comment.as_deref(){
        Some(c) if c==crate::checkpoint_sand::MARKER=>HistoryRequirement::RequiredSandV1,
        Some(c) if c==MARKER=>HistoryRequirement::RequiredV1,
        Some(c) if c.starts_with(b"PPF_PLASTIC_BOUND") && !bound=>return Err("CHECKPOINT_PLASTIC_INVALID: unsupported marker".into()),
        Some(c) if c.starts_with(OUR_PREFIX)=>return Err("CHECKPOINT_HISTORY_MARKER_UNSUPPORTED: unknown or malformed version".into()),
        _=>HistoryRequirement::LegacyUnmarkedAmbiguous,
    };
    let mut payload=Vec::new();
    decoder.by_ref().take(limit.checked_add(1).ok_or("CHECKPOINT_STATE_INVALID: bound overflow")?).read_to_end(&mut payload)
        .map_err(|e|format!("CHECKPOINT_STATE_INVALID: gzip decode: {e}"))?;
    if payload.len() as u64 > limit {return Err("CHECKPOINT_STATE_INVALID: expanded state oversized".into());}
    if !decoder.into_inner().is_empty(){return Err("CHECKPOINT_STATE_INVALID: trailing or concatenated gzip bytes".into());}
    if bound {
        if !allow_plastic_binding || !payload.starts_with(PLASTIC_MAGIC) || payload.len() < 81 {
            return Err("CHECKPOINT_PLASTIC_INVALID: invalid or nested envelope".into());
        }
        let len=u64::from_le_bytes(payload[8..16].try_into().unwrap());
        let hash=std::str::from_utf8(&payload[16..80])
            .map_err(|_| "CHECKPOINT_PLASTIC_INVALID: digest")?;
        if len==0 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("CHECKPOINT_PLASTIC_INVALID: digest or sidecar length".into());
        }
        let mut inner=decode_member(&payload[80..],limit,false)?;
        inner.plastic_binding=Some(Some((len,hash.to_owned())));
        return Ok(inner);
    }
    if payload.starts_with(PLASTIC_MAGIC) {
        return Err("CHECKPOINT_PLASTIC_INVALID: missing plastic marker".into());
    }
    if requirement==HistoryRequirement::RequiredSandV1 {
        if payload.len()<24 || &payload[..8]!=crate::checkpoint_sand::MAGIC {return Err("CHECKPOINT_SAND_INVALID: envelope magic".into());}
        let state_len=u64::from_le_bytes(payload[8..16].try_into().unwrap());
        let sand_len=u64::from_le_bytes(payload[16..24].try_into().unwrap());
        let a=usize::try_from(state_len).map_err(|_|"CHECKPOINT_SAND_INVALID: state length")?;
        let b=usize::try_from(sand_len).map_err(|_|"CHECKPOINT_SAND_INVALID: sand length")?;
        let split=24usize.checked_add(a).ok_or("CHECKPOINT_SAND_INVALID: length overflow")?;
        if b==0 || split.checked_add(b)!=Some(payload.len()) {return Err("CHECKPOINT_SAND_INVALID: envelope length".into());}
        let sand=payload[split..].to_vec();return Ok(DecodedState{requirement,payload:payload[24..split].to_vec(),sand:Some(sand),plastic_binding:None});
    }
    if payload.starts_with(crate::checkpoint_sand::MAGIC) {return Err("CHECKPOINT_SAND_INVALID: missing sand marker".into());}
    Ok(DecodedState{requirement,payload,sand:None,plastic_binding:None})
}

pub(crate) fn bind_plastic(state:&[u8],plastic:Option<&[u8]>)->Result<Vec<u8>,String>{
    let Some(plastic)=plastic else { return Ok(state.to_vec()); };
    let len=(plastic.len() as u64).to_le_bytes();
    let hash=crate::joint::sha256(plastic);
    let mut encoder=GzBuilder::new().comment(PLASTIC_MARKER.to_vec())
        .write(Vec::new(),Compression::default());
    encoder.write_all(PLASTIC_MAGIC).and_then(|_|encoder.write_all(&len))
        .and_then(|_|encoder.write_all(hash.as_bytes()))
        .and_then(|_|encoder.write_all(state)).map_err(|e|e.to_string())?;
    encoder.finish().map_err(|e|e.to_string())
}

pub(crate) fn validate_plastic(binding:&Option<Option<(u64,String)>>,plastic:Option<&[u8]>,scene_plastic:bool)->Result<(),String>{
    match (binding,plastic) {
        (Some(Some((len,hash))),Some(bytes)) if *len==bytes.len() as u64 && *hash==crate::joint::sha256(bytes)=>Ok(()),
        (Some(None),None) | (None,None) if !scene_plastic=>Ok(()),
        (None,_)=>Err("CHECKPOINT_PLASTIC_LEGACY_UNBOUND: plastic checkpoint cannot be verified".into()),
        _=>Err("CHECKPOINT_PLASTIC_MISMATCH: sidecar missing, unexpected, or differs from State binding".into()),
    }
}

/// `validated_history` means the existing sidecar passed content, frame/time,
/// parameter, runtime and generation checks, not merely that a file exists.
/// Preserve other feature-derived requirements by OR-ing them at call site.
pub fn require_validated_history(requirement:HistoryRequirement,validated_history:bool)->Result<(),String>{
    if requirement!=HistoryRequirement::LegacyUnmarkedAmbiguous&&!validated_history {
        Err("CHECKPOINT_HISTORY_REQUIRED: marked State requires validated runtime history".into())
    }else{Ok(())}
}

pub fn encode_state<T:serde::Serialize>(state:&T,history_required:bool)->Result<Vec<u8>,String>{
    let payload=bincode::serialize(state).map_err(|e|format!("CHECKPOINT_STATE_INVALID: {e}"))?;
    let builder=if history_required {GzBuilder::new().comment(MARKER.to_vec())}else{GzBuilder::new()};
    let mut encoder=builder.write(Vec::new(),Compression::default());
    encoder.write_all(&payload).map_err(|e|e.to_string())?;
    encoder.finish().map_err(|e|e.to_string())
}

pub(crate) fn encode_sand_state<T:serde::Serialize>(state:&T,sand:&[u8])->Result<Vec<u8>,String>{
    if sand.is_empty(){return Err("CHECKPOINT_SAND_INVALID: empty grain block".into());}
    let body=bincode::serialize(state).map_err(|e|format!("CHECKPOINT_STATE_INVALID: {e}"))?;
    let size=24usize.checked_add(body.len()).and_then(|v|v.checked_add(sand.len())).ok_or("CHECKPOINT_SAND_INVALID: envelope overflow")?;
    let mut payload=Vec::with_capacity(size);payload.extend_from_slice(crate::checkpoint_sand::MAGIC);
    payload.extend_from_slice(&(body.len() as u64).to_le_bytes());payload.extend_from_slice(&(sand.len() as u64).to_le_bytes());
    payload.extend_from_slice(&body);payload.extend_from_slice(sand);
    let mut encoder=GzBuilder::new().comment(crate::checkpoint_sand::MARKER.to_vec()).write(Vec::new(),Compression::default());
    encoder.write_all(&payload).map_err(|e|e.to_string())?;encoder.finish().map_err(|e|e.to_string())
}

/// Immutable same-frame selection, no writes. The caller MUST separately check
/// existing history semantic equivalence and validate that sidecar before the
/// publisher uses this result. Marker metadata must never trigger migration.
/// Even legacy/alternate-compression bytes are returned EXACTLY as existing.
/// A changed decompressed State is refused, not rewritten or normalized.
pub fn select_immutable_bytes<'a>(existing:Option<&'a [u8]>,candidate:&'a [u8])->Result<Cow<'a,[u8]>,String>{
    let new=decode(candidate)?;
    match existing {
        None=>Ok(Cow::Borrowed(candidate)),
        Some(old_bytes)=>{
            let old=decode(old_bytes)?;
            let binding_ok=old.plastic_binding==new.plastic_binding || (old.plastic_binding.is_none() && new.plastic_binding==Some(None));
            if !binding_ok || old.payload!=new.payload || ((old.sand.is_some() || new.sand.is_some()) && old.sand!=new.sand) {return Err("CHECKPOINT_IMMUTABLE_STATE_MISMATCH: same frame has different State, plastic binding or grain spin".into());}
            Ok(Cow::Borrowed(old_bytes))
        }
    }
}

#[cfg(test)]
mod tests {
 use super::*;
 #[derive(Debug,PartialEq,serde::Serialize,serde::Deserialize)]
 struct MiniState{frame:i32,time:f64,previous_dt:f32,positions:Vec<[f32;3]>}
 fn state()->MiniState{MiniState{frame:22,time:0.224752152338624,previous_dt:0.005,positions:vec![[0.,1.,2.],[3.,4.,5.]]}}
 #[test]fn plastic_binding_accepts_matching_and_refuses_swapped_and_legacy(){
  let raw=encode_state(&state(),true).unwrap();let wrapped=bind_plastic(&raw,Some(b"frame 22")).unwrap();let d=decode(&wrapped).unwrap();
  assert!(validate_plastic(&d.plastic_binding,Some(b"frame 22"),true).is_ok());
  assert!(validate_plastic(&d.plastic_binding,Some(b"frame 23"),true).is_err());
  assert!(validate_plastic(&d.plastic_binding,None,true).is_err());
  assert!(validate_plastic(&decode(&raw).unwrap().plastic_binding,Some(b"frame 22"),true).is_err());
  assert!(validate_plastic(&decode(&raw).unwrap().plastic_binding,None,false).is_ok());
 }
 fn gz(payload:&[u8],comment:Option<&[u8]>,mtime:u32,level:Compression)->Vec<u8>{
  let mut b=GzBuilder::new().mtime(mtime);if let Some(c)=comment{b=b.comment(c.to_vec());}
  let mut e=b.write(Vec::new(),level);e.write_all(payload).unwrap();e.finish().unwrap()
 }
 #[test]fn plastic_binding_preserves_history_and_sand_and_cannot_be_stripped(){
  let raw=encode_state(&state(),false).unwrap();
  assert_eq!(bind_plastic(&raw,None).unwrap(),raw);
  let sand=encode_sand_state(&state(),b"grain state").unwrap();
  for inner in [&raw,&sand] {
   let bytes=bind_plastic(inner,Some(b"frame 22")).unwrap();
   let decoded=decode(&bytes).unwrap();
   let raw_decoded=decode(inner).unwrap();
   assert_eq!(decoded.requirement,raw_decoded.requirement);
   assert_eq!(decoded.sand,raw_decoded.sand);
   assert!(validate_plastic(&decoded.plastic_binding,Some(b"frame 22"),true).is_ok());
   let wrong=bind_plastic(inner,Some(b"frame 23")).unwrap();
   assert!(select_immutable_bytes(Some(&bytes),&wrong).is_err());
   let payload=flate2::read::GzDecoder::new(bytes.as_slice()).bytes()
      .collect::<Result<Vec<_>,_>>().unwrap();
   assert!(decode(&gz(&payload,None,0,Compression::default())).is_err());
   assert!(decode(&bind_plastic(&bytes,Some(b"frame 22")).unwrap()).is_err());
  }
 }
 #[test]fn tagged_and_legacy_preserve_bincode_payload(){
  for marked in [false,true]{let bytes=encode_state(&state(),marked).unwrap();let d=decode(&bytes).unwrap();assert_eq!(d.payload,bincode::serialize(&state()).unwrap());let back:MiniState=bincode::deserialize(&d.payload).unwrap();assert_eq!(back,state());assert_eq!(d.requirement,if marked{HistoryRequirement::RequiredV1}else{HistoryRequirement::LegacyUnmarkedAmbiguous});}
 }
 #[test]fn old_plain_gzip_reader_decodes_marked_state(){
  let bytes=encode_state(&state(),true).unwrap();let mut d=flate2::read::GzDecoder::new(bytes.as_slice());let mut raw=Vec::new();d.read_to_end(&mut raw).unwrap();let back:MiniState=bincode::deserialize(&raw).unwrap();assert_eq!(back,state());
 }
 #[test]fn embedded_sand_state_roundtrip_and_immutable_boundary(){
  let s=state();let data=encode_sand_state(&s,b"accepted-grain").unwrap();let decoded=decode(&data).unwrap();
  assert_eq!(decoded.requirement,HistoryRequirement::RequiredSandV1);
  assert_eq!(decoded.payload,bincode::serialize(&s).unwrap());assert_eq!(decoded.sand.as_deref(),Some(b"accepted-grain".as_slice()));
  assert!(require_validated_history(decoded.requirement,false).is_err());
  assert!(select_immutable_bytes(Some(&data),&encode_sand_state(&s,b"different-grain").unwrap()).is_err());
  assert!(select_immutable_bytes(Some(&encode_state(&s,true).unwrap()),&data).is_err());
  let mut stripped=Vec::new();stripped.extend_from_slice(crate::checkpoint_sand::MAGIC);stripped.extend_from_slice(&0u64.to_le_bytes());stripped.extend_from_slice(&1u64.to_le_bytes());stripped.push(0);
  assert!(decode(&gz(&stripped,None,0,Compression::default())).is_err());
  let mut malformed=stripped;malformed[16..24].copy_from_slice(&100u64.to_le_bytes());
  assert!(decode(&gz(&malformed,Some(crate::checkpoint_sand::MARKER),0,Compression::default())).is_err());
 }
 #[test]fn marker_requires_validated_history(){
  assert!(require_validated_history(HistoryRequirement::RequiredV1,false).unwrap_err().starts_with("CHECKPOINT_HISTORY_REQUIRED"));assert!(require_validated_history(HistoryRequirement::RequiredV1,true).is_ok());
 }
 #[test]fn legacy_is_ambiguous_not_retroactively_protected(){
  let d=decode(&encode_state(&state(),false).unwrap()).unwrap();assert_eq!(d.requirement,HistoryRequirement::LegacyUnmarkedAmbiguous);assert!(require_validated_history(d.requirement,false).is_ok());
 }
 #[test]fn unknown_our_versions_and_malformed_markers_refused(){
  for c in [b"PPF_RUNTIME_HISTORY_REQUIRED_V2".as_slice(),b"PPF_RUNTIME_HISTORY_REQUIRED_V1_extra",b"PPF_RUNTIME_HISTORY_REQUIRED",b"PPF_RUNTIME_HISTORY_REQUIRED_V"]{let bytes=gz(b"x",Some(c),0,Compression::default());assert!(decode(&bytes).unwrap_err().starts_with("CHECKPOINT_HISTORY_MARKER_UNSUPPORTED"));}
 }
 #[test]fn unrelated_comment_remains_legacy_ambiguous(){let d=decode(&gz(b"x",Some(b"another application"),0,Compression::default())).unwrap();assert_eq!(d.requirement,HistoryRequirement::LegacyUnmarkedAmbiguous);}
 #[test]fn truncated_header_and_missing_comment_terminator_refused(){
  let bytes=encode_state(&state(),true).unwrap();for n in 0..=10+MARKER.len(){assert!(decode(&bytes[..n]).is_err(),"accepted prefix length {n}");}
 }
 #[test]fn corrupt_payload_crc_and_truncated_trailer_refused(){
  let bytes=encode_state(&state(),true).unwrap();let mut corrupt=bytes.clone();let n=corrupt.len();corrupt[n-8]^=1;assert!(decode(&corrupt).is_err());for cut in 1..=8 {assert!(decode(&bytes[..bytes.len()-cut]).is_err());}assert!(decode(b"not gzip").is_err());
 }
 #[test]fn same_payload_preserves_exact_legacy_gzip_bytes(){
  let payload=bincode::serialize(&state()).unwrap();let old=gz(&payload,Some(b"legacy arbitrary comment"),1234,Compression::none());let new=encode_state(&state(),true).unwrap();assert_ne!(old,new);let chosen=select_immutable_bytes(Some(&old),&new).unwrap();assert!(matches!(chosen,Cow::Borrowed(_)));assert_eq!(chosen.as_ref(),old.as_slice());assert_eq!(decode(&chosen).unwrap().requirement,HistoryRequirement::LegacyUnmarkedAmbiguous);
 }
 #[test]fn same_payload_preserves_exact_marked_bytes(){let payload=bincode::serialize(&state()).unwrap();let old=gz(&payload,Some(MARKER),1234,Compression::none());let new=encode_state(&state(),true).unwrap();assert_ne!(old,new);assert_eq!(select_immutable_bytes(Some(&old),&new).unwrap().as_ref(),old.as_slice());}
 #[test]fn changed_payload_refused_not_rewritten(){let old=encode_state(&state(),false).unwrap();let mut s=state();s.positions[0][0]=1.;let candidate=encode_state(&s,true).unwrap();assert!(select_immutable_bytes(Some(&old),&candidate).unwrap_err().starts_with("CHECKPOINT_IMMUTABLE_STATE_MISMATCH"));}
 #[test]fn invalid_existing_or_candidate_refused(){let good=encode_state(&state(),true).unwrap();assert!(select_immutable_bytes(Some(b"bad"),&good).is_err());assert!(select_immutable_bytes(Some(&good),b"bad").is_err());}
 #[test]fn first_save_preserves_candidate_bytes(){let good=encode_state(&state(),true).unwrap();assert_eq!(select_immutable_bytes(None,&good).unwrap().as_ref(),good.as_slice());}
 #[test]fn concatenated_or_trailing_data_refused(){let a=encode_state(&state(),true).unwrap();let mut b=a.clone();b.extend_from_slice(&a);assert!(decode(&b).is_err());let mut c=a;c.push(0);assert!(decode(&c).is_err());}
 #[test]fn expanded_bound_enforced(){let bytes=gz(&[0;100],None,0,Compression::default());assert!(decode_bounded(&bytes,99).is_err());assert_eq!(decode_bounded(&bytes,100).unwrap().payload.len(),100);}
}
