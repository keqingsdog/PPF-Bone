//! Host assembly of a body principal block from full-space 3x3 blocks and reduced joint Hessians.
//! Values are column-major 3x3; result is column-major in the supplied free-body order.
#[derive(Clone, Copy)]
pub struct Block { pub row: usize, pub col: usize, pub value: [f32; 9] }
pub struct Joint<'a> { pub ids: &'a [u32], pub h: &'a [f32] }

pub fn assemble(vbody: &[u32], prot: &[f32], bodies: usize, free: &[usize],
                blocks: &[Block], body_base: usize, joints: &[Joint<'_>]) -> Result<Vec<f32>, &'static str> {
    if prot.len()!=vbody.len().checked_mul(3).ok_or("extent")? {return Err("prot extent");}
    let mut slots=vec![None;bodies];
    for (i,&b) in free.iter().enumerate() {if b>=bodies || slots[b].replace(i).is_some(){return Err("free body mapping");}}
    let dim=free.len().checked_mul(6).ok_or("extent")?;
    let mut out=vec![0f64;dim.checked_mul(dim).ok_or("extent")?];
    let basis=|v:usize| -> Result<Option<(usize,[[f64;6];3])>, &'static str> {
        let b=*vbody.get(v).ok_or("vertex index")?;
        if b==u32::MAX{return Ok(None)}
        let b=b as usize;if b>=bodies{return Err("body index")}
        let Some(s)=slots[b] else{return Ok(None)};
        let p=&prot[3*v..3*v+3];if p.iter().any(|x|!x.is_finite()){return Err("nonfinite prot")}
        let (x,y,z)=(p[0] as f64,p[1] as f64,p[2] as f64);
        Ok(Some((s,[[1.,0.,0.,0.,z,-y],[0.,1.,0.,-z,0.,x],[0.,0.,1.,y,-x,0.]])))
    };
    for block in blocks {
        if block.value.iter().any(|v|!v.is_finite()){return Err("nonfinite block")}
        let (Some((a,ja)),Some((b,jb)))=(basis(block.row)?,basis(block.col)?) else{continue};
        // Rows containing structurally nonzero entries in each rotation column.
        const ROT_ROWS: [[usize; 2]; 3] = [[1, 2], [0, 2], [0, 1]];
        for i in 0..6 {for j in 0..6 {
            let mut v=0.;
            if i<3 && j<3 {
                v=block.value[3*j+i] as f64;
            } else if i<3 {
                for &c in &ROT_ROWS[j-3] {v+=(block.value[3*c+i] as f64)*jb[c][j];}
            } else if j<3 {
                for &r in &ROT_ROWS[i-3] {v+=ja[r][i]*(block.value[3*j+r] as f64);}
            } else {
                for &r in &ROT_ROWS[i-3] {for &c in &ROT_ROWS[j-3] {
                    v+=(ja[r][i]*(block.value[3*c+r] as f64))*jb[c][j];
                }}
            }
            out[(6*b+j)*dim+6*a+i]+=v;
        }}
    }
    for joint in joints {
        let n=joint.ids.len();if n>24||joint.h.len()!=n*n{return Err("joint extent")}
        for i in 0..n {for j in 0..n {
            let h=joint.h[n*i+j];if !h.is_finite(){return Err("nonfinite joint")}
            let (a,b)=(joint.ids[i] as usize,joint.ids[j] as usize);
            if a<body_base||b<body_base||a==u32::MAX as usize||b==u32::MAX as usize {continue}
            let (a,b)=(a-body_base,b-body_base);if a>=6*bodies||b>=6*bodies{return Err("joint index")}
            if let (Some(x),Some(y))=(slots[a/6],slots[b/6]) {out[(6*y+b%6)*dim+6*x+a%6]+=h as f64;}
        }}
    }
    out.into_iter().map(|v|{let x=v as f32;if x.is_finite(){Ok(x)}else{Err("nonfinite result")}}).collect()
}

pub struct SparseBodyMatrix { pub body_count:usize, pub free:Vec<usize>, pub blocks:std::collections::BTreeMap<(usize,usize),[[f32;6];6]> }

pub fn assemble_sparse(vbody: &[u32], prot: &[f32], bodies: usize, free: &[usize],
                blocks: &[Block], body_base: usize, joints: &[Joint<'_>]) -> Result<SparseBodyMatrix, &'static str> {
    if prot.len()!=vbody.len().checked_mul(3).ok_or("extent")? {return Err("prot extent");}
    let mut slots=vec![None;bodies];
    for (i,&b) in free.iter().enumerate() {if b>=bodies || slots[b].replace(i).is_some(){return Err("free body mapping");}}
    free.len().checked_mul(6).ok_or("extent")?;
    let mut out=std::collections::BTreeMap::<(usize,usize),[[f64;6];6]>::new();
    let basis=|v:usize| -> Result<Option<(usize,[[f64;6];3])>, &'static str> {
        let b=*vbody.get(v).ok_or("vertex index")?;
        if b==u32::MAX{return Ok(None)}
        let b=b as usize;if b>=bodies{return Err("body index")}
        let Some(s)=slots[b] else{return Ok(None)};
        let p=&prot[3*v..3*v+3];if p.iter().any(|x|!x.is_finite()){return Err("nonfinite prot")}
        let (x,y,z)=(p[0] as f64,p[1] as f64,p[2] as f64);
        Ok(Some((s,[[1.,0.,0.,0.,z,-y],[0.,1.,0.,-z,0.,x],[0.,0.,1.,y,-x,0.]])))
    };
    for block in blocks {
        if block.value.iter().any(|v|!v.is_finite()){return Err("nonfinite block")}
        let (Some((a,ja)),Some((b,jb)))=(basis(block.row)?,basis(block.col)?) else{continue};
        let target=out.entry((a,b)).or_insert([[0.;6];6]);
        // Rows containing structurally nonzero entries in each rotation column.
        const ROT_ROWS: [[usize; 2]; 3] = [[1, 2], [0, 2], [0, 1]];
        for i in 0..6 {for j in 0..6 {
            let mut v=0.;
            if i<3 && j<3 {
                v=block.value[3*j+i] as f64;
            } else if i<3 {
                for &c in &ROT_ROWS[j-3] {v+=(block.value[3*c+i] as f64)*jb[c][j];}
            } else if j<3 {
                for &r in &ROT_ROWS[i-3] {v+=ja[r][i]*(block.value[3*j+r] as f64);}
            } else {
                for &r in &ROT_ROWS[i-3] {for &c in &ROT_ROWS[j-3] {
                    v+=(ja[r][i]*(block.value[3*c+r] as f64))*jb[c][j];
                }}
            }
            target[i][j]+=v;
        }}
    }
    for joint in joints {
        let n=joint.ids.len();if n>24||joint.h.len()!=n*n{return Err("joint extent")}
        for i in 0..n {for j in 0..n {
            let h=joint.h[n*i+j];if !h.is_finite(){return Err("nonfinite joint")}
            let (a,b)=(joint.ids[i] as usize,joint.ids[j] as usize);
            if a<body_base||b<body_base||a==u32::MAX as usize||b==u32::MAX as usize {continue}
            let (a,b)=(a-body_base,b-body_base);if a>=6*bodies||b>=6*bodies{return Err("joint index")}
            if let (Some(x),Some(y))=(slots[a/6],slots[b/6]) {out.entry((x,y)).or_insert([[0.;6];6])[a%6][b%6]+=h as f64;}
        }}
    }
    let mut converted=std::collections::BTreeMap::new(); for (key,block) in out {let mut dst=[[0f32;6];6]; for i in 0..6 {for j in 0..6 {dst[i][j]=block[i][j] as f32;if !dst[i][j].is_finite(){return Err("nonfinite result");}}} converted.insert(key,dst);} Ok(SparseBodyMatrix{body_count:bodies,free:free.to_vec(),blocks:converted})
}

#[cfg(test)] mod tests {
 use super::*;
 #[test] fn projection_matches_dense_oracle_with_roots_and_joints() {
  let vb=[0,0,1,u32::MAX,2];let p=[1.,2.,3., -2.,1.,0.5, 0.,1.,2., 0.,0.,0., 4.,0.,1.];
  let blocks=vec![Block{row:0,col:1,value:[1.,2.,3.,4.,5.,6.,7.,8.,10.]},Block{row:1,col:2,value:[-1.,2.,0.,3.,-2.,1.,5.,4.,1.]},Block{row:2,col:4,value:[1.;9]},Block{row:3,col:0,value:[2.;9]}];
  let ids=[3u32,9,9,15];let h=[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.];
  let got=assemble(&vb,&p,3,&[0,1],&blocks,3,&[Joint{ids:&ids,h:&h}]).unwrap();
  let mut j=vec![vec![0f64;12];15];for v in 0..3 {let b=vb[v] as usize;let (x,y,z)=(p[3*v] as f64,p[3*v+1] as f64,p[3*v+2] as f64);let q=[[1.,0.,0.,0.,z,-y],[0.,1.,0.,-z,0.,x],[0.,0.,1.,y,-x,0.]];for r in 0..3 {for c in 0..6 {j[3*v+r][6*b+c]=q[r][c];}}}
  let mut full=vec![vec![0f64;15];15];for b in &blocks {for r in 0..3 {for c in 0..3 {full[3*b.row+r][3*b.col+c]+=b.value[3*c+r] as f64;}}}
  for a in 0..12 {for b in 0..12 {let mut expected=0.;for r in 0..15 {for c in 0..15 {expected+=j[r][a]*full[r][c]*j[c][b];}}for i in 0..4 {for k in 0..4 {if ids[i] as usize==a+3 && ids[k] as usize==b+3 {expected+=h[4*i+k] as f64;}}}assert!((got[b*12+a] as f64-expected).abs()<1e-5,"{a} {b}");}}
 }
}

#[cfg(test)] mod sparse_tests {
 use super::*;
 #[test] fn sparse_matches_dense_with_roots_fem_and_duplicate_joint_indices() {
  let vb=[0,0,1,u32::MAX,2]; let p=[1.,2.,3.,-2.,1.,0.5,0.,1.,2.,0.,0.,0.,4.,0.,1.];
  let blocks=[Block{row:0,col:1,value:[1.,2.,3.,4.,5.,6.,7.,8.,10.]},Block{row:1,col:2,value:[-1.,2.,0.,3.,-2.,1.,5.,4.,1.]},Block{row:2,col:4,value:[1.;9]},Block{row:3,col:0,value:[2.;9]}];
  let ids=[3u32,9,9,15]; let h=[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]; let joints=[Joint{ids:&ids,h:&h}];
  let dense=assemble(&vb,&p,3,&[1,0],&blocks,3,&joints).unwrap(); let sparse=assemble_sparse(&vb,&p,3,&[1,0],&blocks,3,&joints).unwrap();
  for a in 0..2 {for b in 0..2 {for i in 0..6 {for j in 0..6 {let x=sparse.blocks.get(&(a,b)).map_or(0.,|v|v[i][j]);assert_eq!(x.to_bits(),dense[(6*b+j)*12+6*a+i].to_bits());}}}}
 }
 #[test] fn sparse_rejects_nonfinite_even_when_body_is_fixed() {
  let blocks=[Block{row:0,col:1,value:[f32::NAN;9]}];
  assert!(assemble_sparse(&[0,1],&[0.;6],2,&[0],&blocks,0,&[]).is_err());
 }
}


