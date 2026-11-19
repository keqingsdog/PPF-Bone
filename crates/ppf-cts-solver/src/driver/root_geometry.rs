//! Read-only proper rigid measurement of actual native vertex geometry.
//! No target/GPU pose input, physical buffer write, projection or clock change.
use nalgebra::{Matrix3,UnitQuaternion,Vector3,linalg::SVD};
use std::collections::HashSet;

#[derive(Clone,Debug)]
pub(crate) struct GeometryPose {
    pub center:[f64;3],
    pub rotation:[f64;9],
    pub quaternion:[f64;4],
    pub count:u32,
    pub radius:f64,
    pub max_component_residual:f64,
}

/// Safe adapters for the native scene vector and staged packed xyz buffers.
/// Components are copied without arithmetic; fit_body retains finite/domain
/// validation of the actual selected body. No layout reinterpretation.
pub(crate) fn rest_vectors(rest:&[Vector3<f32>])->Vec<[f32;3]> {
    rest.iter().map(|v|[v[0],v[1],v[2]]).collect()
}
pub(crate) fn rest_flat(rest:&[f32])->Result<Vec<[f32;3]>,String> {
    if rest.len()%3!=0 {return Err("invalid packed rest xyz namespace".into());}
    Ok(rest.chunks_exact(3).map(|v|[v[0],v[1],v[2]]).collect())
}

fn finite_vector(v:&Vector3<f64>)->bool {v.iter().all(|v|v.is_finite())}
fn finite_matrix(v:&Matrix3<f64>)->bool {v.iter().all(|v|v.is_finite())}
fn identifiable_rank(m:&Matrix3<f64>)->Result<(),String> {
    if !finite_matrix(m) {return Err("nonfinite geometry Gram/covariance".into());}
    let s=SVD::try_new(*m,false,false,5.0*f64::EPSILON,128).ok_or("geometry rank SVD did not converge")?.singular_values;
    if !s.iter().all(|v|v.is_finite()) {return Err("nonfinite geometry rank evidence".into());}
    let largest=s.iter().copied().fold(0.0_f64,f64::max);
    let mut sorted=[s[0],s[1],s[2]];sorted.sort_by(|a,b|b.total_cmp(a));
    // Roundoff bound only: a numerically unresolved direction is refused;
    // this is independent of the physical/member/angular tolerances.
    // Rank two is sufficient: two noncollinear axes determine the SO(3)
    // normal. Do not reject existing valid planar rigid frame geometry.
    if largest<=0.0 || sorted[1]<=128.0*f64::EPSILON*largest {
        return Err("actual/rest geometry lacks two resolvable independent axes".into());
    }
    Ok(())
}

pub(crate) fn fit_body(positions:&[f32],rest:&[[f32;3]],list:&[u32],
    body:u32,start:u32,count:u32,position_scale:f64,body_of_vertex:impl Fn(u32)->Option<u32>) -> Result<GeometryPose,String> {
    if !position_scale.is_finite()||position_scale<=0.0 {return Err("invalid geometry unit scale".into());}
    if positions.len()%3!=0 || count<4 {return Err("invalid actual geometry count/xyz namespace".into());}
    let end=start.checked_add(count).ok_or("geometry span overflow")? as usize;
    let members=list.get(start as usize..end).ok_or("geometry membership span absent")?;
    let ys=rest.get(start as usize..end).ok_or("geometry rest span absent")?;
    let expected_body=body.checked_add(1).ok_or("geometry body index overflow")?;
    let namespace=u32::try_from(positions.len()/3).map_err(|_|"geometry vertex namespace exceeds u32")?;
    let mut native_count=0_u32;
    for v in 0..namespace {
        let member=body_of_vertex(v).ok_or("geometry native body membership absent")?;
        if member==expected_body {native_count=native_count.checked_add(1).ok_or("geometry member count overflow")?;}
    }
    if native_count!=count {return Err("geometry span does not cover the full native body membership".into());}
    let mut seen=HashSet::new();let mut xx=Vec::with_capacity(count as usize);let mut yy=Vec::with_capacity(count as usize);
    for (&v,y) in members.iter().zip(ys) {
        if !seen.insert(v) || body_of_vertex(v)!=Some(expected_body) {return Err("geometry duplicate/wrong body member".into());}
        let at=(v as usize).checked_mul(3).ok_or("geometry vertex offset overflow")?;
        let end=at.checked_add(3).ok_or("geometry vertex range overflow")?;
        let x=positions.get(at..end).ok_or("geometry vertex absent")?;
        let x=Vector3::new(f64::from(x[0]),f64::from(x[1]),f64::from(x[2]))*position_scale;
        let y=Vector3::new(f64::from(y[0]),f64::from(y[1]),f64::from(y[2]));
        if !finite_vector(&x)||!finite_vector(&y) {return Err("nonfinite actual/rest geometry".into());}
        xx.push(x);yy.push(y);
    }
    // Anchor-relative mean avoids summing large world offsets. Retain f64.
    let (ax,ay)=(xx[0],yy[0]);let mut sx=Vector3::zeros();let mut sy=Vector3::zeros();
    for (x,y) in xx.iter().zip(&yy) {
        sx+=x-ax;sy+=y-ay;
        if !finite_vector(&sx)||!finite_vector(&sy) {return Err("geometry mean sum overflow".into());}
    }
    let mx=ax+sx/f64::from(count);let my=ay+sy/f64::from(count);
    if !finite_vector(&mx)||!finite_vector(&my) {return Err("nonfinite geometry mean".into());}
    let mut cov=Matrix3::zeros();let mut gx=Matrix3::zeros();let mut gy=Matrix3::zeros();let mut radius=0.0_f64;
    for (x,y) in xx.iter().zip(&yy) {
        let xc=x-mx;let yc=y-my;
        cov+=yc*xc.transpose();gx+=xc*xc.transpose();gy+=yc*yc.transpose();radius=radius.max(y.norm());
        if !finite_matrix(&cov)||!finite_matrix(&gx)||!finite_matrix(&gy)||!radius.is_finite() {return Err("geometry covariance sum overflow".into());}
    }
    identifiable_rank(&gx)?;identifiable_rank(&gy)?;identifiable_rank(&cov)?;
    let svd=SVD::try_new(cov,true,true,5.0*f64::EPSILON,128).ok_or("geometry pose SVD did not converge")?;let u=svd.u.ok_or("geometry SVD U absent")?;let vt=svd.v_t.ok_or("geometry SVD V absent")?;
    if !finite_matrix(&u)||!finite_matrix(&vt)||!svd.singular_values.iter().all(|v|v.is_finite()) {return Err("nonfinite geometry SVD".into());}
    let v=vt.transpose();let sign=(v*u.transpose()).determinant();
    if !sign.is_finite() || sign.abs()<0.5 {return Err("invalid geometry SVD orientation".into());}
    if sign<0.0 && svd.singular_values[2]>128.0*f64::EPSILON*svd.singular_values[0]
        && svd.singular_values[1]-svd.singular_values[2]<=128.0*f64::EPSILON*svd.singular_values[0] {
        return Err("reflected geometry has no unique proper rotation".into());
    }
    let r=v*Matrix3::from_diagonal(&Vector3::new(1.0,1.0,if sign<0.0 {-1.0}else{1.0}))*u.transpose();
    let center=mx-r*my;
    if !finite_vector(&center)||!finite_matrix(&r)||(r.transpose()*r-Matrix3::identity()).norm()>1e-4||(r.determinant()-1.0).abs()>1e-4 {
        return Err("actual geometry fit is not finite/proper".into());
    }
    let q=UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(r));let q=q.quaternion();let quaternion=[q.w,q.i,q.j,q.k];
    if !quaternion.iter().all(|v|v.is_finite()) || (quaternion.iter().map(|v|v*v).sum::<f64>()-1.0).abs()>1e-6 {return Err("invalid actual geometry quaternion".into());}
    let mut max_component_residual=0.0_f64;
    for (x,y) in xx.iter().zip(&yy) {
        let residual=x-(center+r*y);
        if !finite_vector(&residual) {return Err("geometry reconstruction overflow".into());}
        max_component_residual=max_component_residual.max(residual.iter().map(|v|v.abs()).fold(0.0,f64::max));
    }
    Ok(GeometryPose{center:center.into(),rotation:std::array::from_fn(|i|r.as_slice()[i]),quaternion,count,radius,max_component_residual})
}
