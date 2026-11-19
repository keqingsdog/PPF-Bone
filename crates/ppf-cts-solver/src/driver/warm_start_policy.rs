/// The caller forms Ad from two current affine applications. The operator
/// must retain positive curvature; no non-finite scalar may reach a candidate.
pub fn alpha(numerator: f32, denominator: f32, curvature: f32) -> Option<f32> {
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 0.0
        || !curvature.is_finite() || curvature <= 0.0 { return None; }
    let alpha = numerator / denominator;
    alpha.is_finite().then_some(alpha)
}

/// This uses the original cold per-group scales, not the global L2 norm.
pub fn accept(candidate_worst: f32, cold_worst: f32) -> bool {
    candidate_worst.is_finite() && cold_worst.is_finite() && candidate_worst < cold_worst
}

/// Solve the current energy projection, refusing nonpositive curvature and
/// dropping the older direction if the 2D matrix is nearly singular.
pub fn energy_coefficients(g11:f32,g12:f32,g21:f32,g22:f32,b1:f32,b2:f32,have_two:bool) -> Option<(f32,f32,u8)> {
    if !g11.is_finite() || g11 <= 0.0 || !b1.is_finite() { return None; }
    let single = || { let c=b1/g11; c.is_finite().then_some((c,0.0,1)) };
    if !have_two || !g12.is_finite() || !g21.is_finite() || !g22.is_finite() || g22 <= 0.0 || !b2.is_finite() { return single(); }
    let (a,b,c,d,e)=(f64::from(g11),f64::from(g12),f64::from(g21),f64::from(g22),f64::from(b1));
    let scale=(a*d).sqrt();
    if (b-c).abs()>1e-3*scale { return single(); }
    let off=(b+c)*0.5;
    let det=a*d-off*off;
    if !det.is_finite() || det<=1e-5*a*d { return single(); }
    let f=f64::from(b2);
    let x=((e*d-off*f)/det) as f32;
    let y=((a*f-off*e)/det) as f32;
    if x.is_finite() && y.is_finite() {Some((x,y,2))} else {single()}
}

pub fn accept_energy(delta_energy:f32, candidate_worst:f32) -> bool {
    delta_energy.is_finite() && delta_energy < 0.0 && candidate_worst.is_finite() && candidate_worst <= 2.0
}

/// Scheduling threshold for deciding whether the two-direction setup is
/// worth attempting, measured only from verified cold solves of this key.
const AUTO_COST_MIN_ITERATIONS:u32 = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColdCost { recent:[u32;2], count:u8 }
impl ColdCost {
    pub fn for_key(enabled:bool, previous:Option<&WarmKey>, current:Option<&WarmKey>, prior:Self)->Self {
        if enabled && previous.is_some() && previous==current {prior} else {Self::default()}
    }
    pub fn allow_candidate(&self)->bool {
        self.count==2 && self.recent[0].max(self.recent[1])>=AUTO_COST_MIN_ITERATIONS
    }
    pub fn record_success(&mut self, iterations:u32, true_verified:bool, warm_accepted:bool) {
        if !true_verified || warm_accepted {return;}
        if self.count<2 {self.recent[self.count as usize]=iterations;self.count+=1;}
        else {self.recent=[self.recent[1],iterations];}
    }
    pub fn samples(&self)->(u8,[u32;2]) {(self.count,self.recent)}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_offset_cancels_and_negative_alpha_is_permitted() {
        // A_aff(x) = diag(2,3)x + (5,-4), cold=(1,2), h=(1,-1).
        let affine = |x: [f32;2]| [2.0*x[0]+5.0, 3.0*x[1]-4.0];
        let cold=affine([1.0,2.0]); let plus=affine([2.0,1.0]);
        let ad=[plus[0]-cold[0],plus[1]-cold[1]];
        assert_eq!(ad,[2.0,-3.0]);
        let r=[-2.0,3.0];
        let a=alpha(r[0]*ad[0]+r[1]*ad[1],ad[0]*ad[0]+ad[1]*ad[1],5.0).unwrap();
        assert_eq!(a,-1.0);
        assert_eq!([r[0]-a*ad[0],r[1]-a*ad[1]],[0.0,0.0]);
    }

    #[test]
    fn invalid_curvature_and_nonfinite_scalars_reject() {
        for (n,d,c) in [(1.0,0.0,1.0),(1.0,1.0,0.0),(f32::NAN,1.0,1.0),
                        (1.0,f32::INFINITY,1.0),(1.0,1.0,f32::NAN)] {
            assert_eq!(alpha(n,d,c), None);
        }
    }

    #[test]
    fn worst_group_must_strictly_improve() {
        assert!(accept(0.9,1.0));
        assert!(!accept(1.0,1.0));
        assert!(!accept(1.2,1.0));
        assert!(!accept(f32::NAN,1.0));
    }

    #[test]
    fn cache_key_compares_exact_mapping_mask_and_timestep() {
        let key=WarmKey {nrow:2,n_cloth:2,bodies:0,dim:6,body_base:6,
            vbody:vec![u32::MAX,u32::MAX],cloth_off:vec![0,3],mask:vec![0,0],dt_bits:0.01f32.to_bits(),mode:2};
        let mut previous=Some(key.clone());
        let captured=previous.take();
        assert!(previous.is_none()); // any non-successful exit leaves it invalid
        assert_eq!(captured.as_ref(),Some(&key));
        let mut changed=key.clone(); changed.cloth_off.swap(0,1); assert_ne!(captured.as_ref(),Some(&changed));
        changed=key.clone(); changed.vbody[0]=0; assert_ne!(captured.as_ref(),Some(&changed));
        changed=key.clone(); changed.mask[1]=1; assert_ne!(captured.as_ref(),Some(&changed));
        changed=key.clone(); changed.dt_bits=0.02f32.to_bits(); assert_ne!(captured.as_ref(),Some(&changed));
        changed=key.clone(); changed.mode=1; assert_ne!(captured.as_ref(),Some(&changed));
    }

    #[test]
    fn nonzero_vertex_mask_maps_to_fixed_reduced_triples_only() {
        use super::super::super::rigid_map::{RigidMap, RIGID_UNSET};
        let map=RigidMap { nrow:3,n_bodies:1,n_cloth:2,dim:12,body_base:6,
            vbody:vec![RIGID_UNSET,RIGID_UNSET,0],cloth_off:vec![0,3,0],..RigidMap::default() };
        let free=reduced_free_rows(&map,&[1,0,0]).unwrap();
        assert_eq!(&free[0..3],&[0,0,0]);
        assert_eq!(&free[3..6],&[1,1,1]);
        assert_eq!(&free[6..12],&[1,1,1,1,1,1]);
        assert!(reduced_free_rows(&map,&[0,0,1]).is_none());
    }

    #[test]
    fn identical_nonzero_mask_key_is_reusable_but_a_changed_mask_is_not() {
        let key=WarmKey {nrow:2,n_cloth:2,bodies:0,dim:6,body_base:6,
            vbody:vec![u32::MAX,u32::MAX],cloth_off:vec![0,3],mask:vec![1,0],dt_bits:0.01f32.to_bits(),mode:1};
        assert_eq!(Some(&key),Some(&key.clone()));
        let mut changed=key.clone(); changed.mask[1]=1;
        assert_ne!(Some(&key),Some(&changed));
        let map=super::super::super::rigid_map::RigidMap {nrow:2,n_cloth:2,dim:6,body_base:6,
            vbody:vec![u32::MAX,u32::MAX],cloth_off:vec![0,3],..Default::default()};
        assert!(reduced_free_rows(&map,&key.mask).is_some());
        assert!(reduced_free_rows(&map,&changed.mask).is_some());
        assert_eq!(ColdCost::for_key(true,Some(&key),Some(&key),ColdCost::default()).samples().0,0);
        assert_eq!(ColdCost::for_key(true,Some(&key),Some(&changed),ColdCost::default()).samples().0,0);
    }

    #[test]
    fn fixed_history_rows_leave_the_current_affine_seed_unchanged() {
        let base=[2.0f32,3.0,4.0,5.0,6.0,7.0];
        let mut history=[11.0f32,12.0,13.0,14.0,15.0,16.0];
        let free=[0u32,0,0,1,1,1];
        for i in 0..history.len() { if free[i]==0 { history[i]=0.0; } }
        let candidate:Vec<_>=base.iter().zip(history).map(|(a,b)|a+b).collect();
        assert_eq!(&candidate[0..3],&base[0..3]);
        assert_eq!(&candidate[3..6],&[19.0,21.0,23.0]);
        // The existing strict residual gate and tolerance remain the only
        // convergence decision; masking changes candidate rows, not the rule.
        assert!(accept(0.9,1.0));
        assert!(!accept(1.0,1.0));
    }

    #[test]
    fn energy_projection_and_rank_deficiency() {
        let (x,y,rank)=energy_coefficients(4.0,1.0,1.0,3.0,1.0,2.0,true).unwrap();
        assert_eq!(rank,2);
        assert!((4.0*x+y-1.0).abs()<1e-6 && (x+3.0*y-2.0).abs()<1e-6);
        assert_eq!(energy_coefficients(4.0,2.0,2.0,1.0,8.0,4.0,true),Some((2.0,0.0,1)));
        assert_eq!(energy_coefficients(-1.0,0.0,0.0,1.0,1.0,1.0,true),None);
        assert_eq!(energy_coefficients(f32::NAN,0.0,0.0,1.0,1.0,1.0,true),None);
    }

    #[test]
    fn energy_and_original_group_budget_both_required() {
        assert!(accept_energy(-0.1,2.0));
        assert!(!accept_energy(0.0,0.1));
        assert!(!accept_energy(-0.1,2.01));
        assert!(!accept_energy(f32::NAN,0.1));
        assert!(!accept_energy(-0.1,f32::INFINITY));
        // A_aff(x)=2x+5, cold=1, d=2: the affine constant cancels.
        let (cold,delta,r0)=(1.0f32,2.0f32,5.0f32);
        let affine=|x:f32| 2.0*x+5.0;
        let ad=affine(cold+delta)-affine(cold);
        let actual_r=r0-ad;
        let energy=-0.5*delta*(r0+actual_r);
        assert_eq!(energy,-delta*r0+0.5*delta*ad);
    }

    #[test]
    fn auto_cost_learns_cold_only_and_recovers_after_short_skip() {
        let mut cost=ColdCost::default();assert!(!cost.allow_candidate());
        cost.record_success(2,true,true);assert_eq!(cost.samples().0,0); // warm is never cold evidence
        cost.record_success(9,false,false);assert_eq!(cost.samples().0,0); // no true verification
        cost.record_success(8,true,false);assert!(!cost.allow_candidate());
        cost.record_success(31,true,false);assert!(!cost.allow_candidate());
        // Even a skipped solve updates the cold observations.
        cost.record_success(32,true,false);assert!(cost.allow_candidate());
        cost.record_success(1,true,true);assert_eq!(cost.samples(),(2,[31,32]));
        // A rejected candidate that finishes from cold counts as cold work.
        cost.record_success(7,true,false);assert!(cost.allow_candidate());
        cost.record_success(6,true,false);assert!(!cost.allow_candidate());
    }

    #[test]
    fn auto_cost_uses_exact_key_and_disabled_mode_does_not_reuse_evidence() {
        let key=WarmKey {nrow:1,n_cloth:1,bodies:0,dim:3,body_base:3,
            vbody:vec![u32::MAX],cloth_off:vec![0],mask:vec![0],dt_bits:0.01f32.to_bits(),mode:2};
        let mut cost=ColdCost::default();cost.record_success(33,true,false);cost.record_success(40,true,false);
        assert!(ColdCost::for_key(true,Some(&key),Some(&key),cost).allow_candidate());
        let mut changed=key.clone();changed.dt_bits=0.02f32.to_bits();
        assert_eq!(ColdCost::for_key(true,Some(&key),Some(&changed),cost).samples().0,0);
        assert_eq!(ColdCost::for_key(true,None,None,cost).samples().0,0);
        assert_eq!(ColdCost::for_key(false,Some(&key),Some(&key),cost).samples().0,0);
    }
}
#[derive(Clone, PartialEq, Debug)]
pub struct WarmKey {
    pub nrow: usize, pub n_cloth: usize, pub bodies: usize, pub dim: usize, pub body_base: usize,
    pub vbody: Vec<u32>, pub cloth_off: Vec<u32>, pub mask: Vec<u32>, pub dt_bits: u32, pub mode:u8,
}

/// Build a scalar mask for homogeneous reduced warm corrections. The vertex
/// mask uses nonzero for prescribed/removal rows; only unbodied vertices map
/// to reduced triples, while each rigid body's six coordinates remain free.
pub fn reduced_free_rows(map: &super::super::rigid_map::RigidMap, vertex_mask: &[u32]) -> Option<Vec<u32>> {
    if vertex_mask.len() != map.nrow
        || map.vbody.len() < map.nrow
        || map.cloth_off.len() < map.nrow
        || map.body_base != map.n_cloth.checked_mul(3)?
        || map.dim != map.body_base.checked_add(map.n_bodies.checked_mul(6)?)?
    {
        return None;
    }
    let mut free = vec![1u32; map.dim];
    let mut seen = vec![false; map.n_cloth];
    for vertex in 0..map.nrow {
        if map.vbody[vertex] != super::super::rigid_map::RIGID_UNSET {
            // The producer only marks free cloth vertices; a fixed body vertex
            // would have no corresponding reduced scalar rows.
            if vertex_mask[vertex] != 0 { return None; }
            continue;
        }
        let offset = map.cloth_off[vertex] as usize;
        if offset % 3 != 0 || offset.checked_add(3)? > map.body_base { return None; }
        let compact = offset / 3;
        if compact >= map.n_cloth || seen[compact] { return None; }
        seen[compact] = true;
        if vertex_mask[vertex] != 0 { free[offset..offset + 3].fill(0); }
    }
    if seen.iter().any(|&mapped| !mapped) { return None; }
    Some(free)
}

