//! Field-only pin transitions over live device element records.
//! No rest/plastic/material byte is uploaded from the host snapshot.
use ppf_cts_compute::{Device,Fault,Pod,StagedBuffer};
use crate::data::{FaceProp,EdgeProp,TetProp,HingeProp};

pub(super) trait FixedRecord:Pod+Default {
    const FIXED_OFFSET:usize;
    fn fixed(&self)->bool;
}
macro_rules! record {
    ($t:ty)=>{impl FixedRecord for $t {
        const FIXED_OFFSET:usize=std::mem::offset_of!($t,fixed);
        fn fixed(&self)->bool{self.fixed}
    }};
}
record!(FaceProp);record!(EdgeProp);record!(TetProp);record!(HingeProp);

fn bad(detail:&str)->Fault {Fault::Platform{call:"element fixed flags",detail:detail.into()}}
fn validate<T:FixedRecord>(buffer:&StagedBuffer<T>,records:&[T])->Result<(),Fault>{
    if buffer.len()!=records.len(){return Err(bad("fixed record count differs from device allocation"));}
    buffer.validate_bool_field_patch(T::FIXED_OFFSET)
}
fn patch<T:FixedRecord,D:Device>(device:&mut D,buffer:&mut StagedBuffer<T>,records:&[T])->Result<usize,Fault>{
    let mut changed=0;
    for (index,record) in records.iter().enumerate(){
        // Safety: repr(C) production record, actual offset_of its bool field.
        changed+=usize::from(unsafe {buffer.patch_bool_field(device,index,T::FIXED_OFFSET,record.fixed())?});
    }
    Ok(changed)
}
/// Validate every allocation before the first write; failures are fatal to the
/// caller. A failed transfer invalidates that buffer's dispatchable clean state.
pub(super) fn restage<D:Device>(device:&mut D,
    faces:&mut StagedBuffer<FaceProp>,face_records:&[FaceProp],
    edges:&mut StagedBuffer<EdgeProp>,edge_records:&[EdgeProp],
    tets:&mut StagedBuffer<TetProp>,tet_records:&[TetProp],
    hinges:&mut StagedBuffer<HingeProp>,hinge_records:&[HingeProp],
)->Result<usize,Fault>{
    validate(faces,face_records)?;validate(edges,edge_records)?;
    validate(tets,tet_records)?;validate(hinges,hinge_records)?;
    let result=(|| {
        let mut changed=patch(device,faces,face_records)?;
        changed+=patch(device,edges,edge_records)?;
        changed+=patch(device,tets,tet_records)?;
        changed+=patch(device,hinges,hinge_records)?;
        Ok(changed)
    })();
    if result.is_err(){faces.invalidate_clean_state();edges.invalidate_clean_state();tets.invalidate_clean_state();hinges.invalidate_clean_state();}
    result
}

/// Streamed rest schedules own only these exclusion bits, independently of pins.
/// This preserves the preexisting update_rest_shape path without whole records.
pub(super) fn restage_rest_excluded<D:Device>(device:&mut D,
    faces:&mut StagedBuffer<FaceProp>,face_records:&[FaceProp],
    tets:&mut StagedBuffer<TetProp>,tet_records:&[TetProp],
)->Result<usize,Fault>{
    if faces.len()!=face_records.len()||tets.len()!=tet_records.len(){return Err(bad("rest exclusion count differs from device allocation"));}
    let fo=std::mem::offset_of!(FaceProp,rest_excluded);
    let to=std::mem::offset_of!(TetProp,rest_excluded);
    faces.validate_bool_field_patch(fo)?;tets.validate_bool_field_patch(to)?;
    let result=(|| {
        let mut changed=0;
        for (i,r) in face_records.iter().enumerate(){changed+=usize::from(unsafe{faces.patch_bool_field(device,i,fo,r.rest_excluded)?});}
        for (i,r) in tet_records.iter().enumerate(){changed+=usize::from(unsafe{tets.patch_bool_field(device,i,to,r.rest_excluded)?});}
        Ok(changed)
    })();
    if result.is_err(){faces.invalidate_clean_state();tets.invalidate_clean_state();}
    result
}
