// Pure decision logic. Explicit controls opt out of the combined automatic profile.
#[derive(Clone, Copy, Debug)]
pub struct Domain {
    pub cuda: bool, pub fp32: bool, pub newton: bool, pub bodies: bool,
    pub joints: bool, pub compatible: bool, pub explicit: bool,
}
pub fn candidate(d: Domain) -> bool {
    d.cuda && d.fp32 && d.newton && d.bodies && d.joints && d.compatible && !d.explicit
}
pub fn automatic_active(candidate: bool, prepared: bool) -> bool { candidate && prepared }
#[cfg(test)] mod tests {
    use super::*;
    fn supported() -> Domain { Domain {cuda:true,fp32:true,newton:true,bodies:true,joints:true,compatible:true,explicit:false} }
    #[test] fn automatic_only_for_supported_domain() {
        assert!(candidate(supported()));
        let mut d=supported(); d.cuda=false; assert!(!candidate(d));
        let mut d=supported(); d.fp32=false; assert!(!candidate(d));
        let mut d=supported(); d.newton=false; assert!(!candidate(d));
        let mut d=supported(); d.bodies=false; assert!(!candidate(d));
        let mut d=supported(); d.joints=false; assert!(!candidate(d));
        let mut d=supported(); d.compatible=false; assert!(!candidate(d));
    }
    #[test] fn explicit_control_and_failed_preparation_do_not_activate_auto() {
        let mut d=supported(); d.explicit=true; assert!(!candidate(d));
        assert!(!automatic_active(true,false));
        assert!(automatic_active(true,true));
        assert!(!automatic_active(false,true));
    }
}


pub fn explicit_controls() -> bool { explicit_controls_with(|k| std::env::var_os(k).is_some()) }

pub fn explicit_controls_with(has: impl Fn(&str) -> bool) -> bool {
    ["PPF_BODY_TREE_PCG", "PPF_BODY_TREE_APPLY", "PPF_BODY_TREE_FACTOR_REUSE",
     "PPF_JOINT_RESIDENT_PCG", "PPF_JOINT_RESIDENT_ROOT", "PPF_JOINT_FROZEN_TANGENT",
     "PPF_PCG_COALESCE_MIXED", "PPF_JOINT_ROW_PARALLEL", "PPF_PCG_WARM_START", "PPF_JOINT_FUSED_PRODUCT",
     "PPF_JOINT_RESIDENT_BATCH8", "PPF_BODY_TREE_LAZY", "PPF_BODY_TREE_DEBUG_RESTART",
     "PPF_BODY_GROUP_SOLVE", "PPF_BODY_SCALAR64_SOLVE",
     "PPF_PCG_INERTIAL_SEED", "PPF_PCG_INERTIAL_PROBE", "PPF_PCG_ANIMATION_MOTION_TOL",
     "PPF_REDUCED_OPERATOR_SHADOW", "PPF_TREE_SHADOW", "PPF_TREE_DIRECT_USE",
     "PPF_RESTRICT_GATHER", "PPF_TET_FORCE_ONLY",
     "PPF_PCG_TRUE_RESIDUAL_CAPTURE_DIR", "PPF_RIGID_PCG_FAILURE_CAPTURE_DIR",
     "PPF_PCG_TRUE_RESIDUAL_DIAGNOSTICS", "PPF_PCG_TRUE_RESIDUAL_CAPTURE_STOP"].iter().any(|k| has(k))
}



#[cfg(test)] mod override_tests {
    use super::*;
    #[test] fn explicit_combination_flags_disable_automatic_profile() {
        for name in ["PPF_BODY_TREE_PCG", "PPF_BODY_TREE_APPLY", "PPF_BODY_TREE_FACTOR_REUSE", "PPF_JOINT_RESIDENT_PCG", "PPF_JOINT_RESIDENT_ROOT", "PPF_JOINT_FROZEN_TANGENT", "PPF_JOINT_ROW_PARALLEL", "PPF_PCG_COALESCE_MIXED"] {
            assert!(explicit_controls_with(|key| key == name));
        }
        assert!(!explicit_controls_with(|key| key == "PPF_PCG_TIMING_RECEIPT"));
        for name in ["PPF_PCG_INERTIAL_SEED", "PPF_PCG_INERTIAL_PROBE", "PPF_PCG_TRUE_RESIDUAL_CAPTURE_DIR", "PPF_RIGID_PCG_FAILURE_CAPTURE_DIR", "PPF_REDUCED_OPERATOR_SHADOW", "PPF_TREE_SHADOW", "PPF_TREE_DIRECT_USE", "PPF_RESTRICT_GATHER", "PPF_TET_FORCE_ONLY"] { assert!(explicit_controls_with(|key| key == name)); }
    }
}




