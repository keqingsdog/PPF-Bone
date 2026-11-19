//! Read-only closest proper material rotation of the actual deformation.
use nalgebra::Matrix3;

pub(crate) fn proper_rotation(f: &Matrix3<f64>) -> Result<Matrix3<f64>, String> {
    if !f.iter().all(|v| v.is_finite()) || !f.determinant().is_finite() {
        return Err("nonfinite material frame".into());
    }
    let svd=f.svd(true,true);
    let singular=svd.singular_values;
    if !singular.iter().all(|v|v.is_finite() && *v>=0.) {return Err("invalid material singular values".into());}
    let mut u=svd.u.ok_or("missing material SVD U")?;
    let vt=svd.v_t.ok_or("missing material SVD Vt")?;
    let uv=u*vt;
    let orientation=uv.determinant();
    if !orientation.is_finite() || (orientation.abs()-1.).abs()>5e-4 {return Err("invalid material SVD orientation".into());}
    let mut order=[0usize,1,2];order.sort_by(|&a,&b|singular[a].total_cmp(&singular[b]));
    let mid=singular[order[1]];let hi=singular[order[2]];
    if !(hi>0. && mid>2e-6*hi) {return Err("material rotation is rank deficient".into());}
    // Preserve the established full-rank positive branch exactly.
    if f.determinant()>0. && singular.iter().all(|v|*v>0.) {return Ok(uv);}
    if orientation<0. {for i in 0..3 {u[(i,order[0])]= -u[(i,order[0])];}}
    let r=u*vt;
    let stretch=r.transpose()*f;
    let maxs=stretch.amax();
    if !maxs.is_finite() || maxs<=0. || !(stretch-stretch.transpose()).amax().is_finite()
        || (stretch-stretch.transpose()).amax()>2e-5*maxs {
        return Err("invalid signed material stretch".into());
    }
    let h=(stretch+stretch.transpose())*(0.5/maxs);
    let k=Matrix3::identity()*h.trace()-h;
    // Same scale-relative K certificate as the device fallback. In
    // particular, reflected matrices with equal two smallest singular
    // values have no unique closest proper rotation.
    if !(k[(0,0)]>1e-6 && k[(0,0)]*k[(1,1)]-k[(0,1)]*k[(1,0)]>1e-6 && k.determinant()>1e-6) {
        return Err("material rotation is non-unique or ill-conditioned".into());
    }
    Ok(r)
}
