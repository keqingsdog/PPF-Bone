//! Mesh sweep-certificate eligibility for opt-in shadow comparison.

/// A certificate is local to its issuing sweep and matching accept scope (B29 to B31, or B32).
/// It must not cross Newton iterations, retries, or be exchanged between scopes.
/// `raw_toi` is the unnormalized mesh CCD time, independent of analytic TOI.
#[derive(Clone, Copy, Debug)]
pub struct SweepCertificate {
    pub raw_toi: f32,
    pub ceiling: f32,
    pub overlapping_start: bool,
}

impl SweepCertificate {
    pub fn covers_endpoint(self) -> bool {
        !self.overlapping_start && self.ceiling.is_finite() && self.ceiling >= 1.0
            && self.raw_toi.is_finite() && self.raw_toi >= 1.0 && self.raw_toi <= self.ceiling
    }
}

/// Compare initialized native-endian FP32 coordinates bitwise, including signed
/// zero. Reject all NaNs, infinities and malformed/incomplete byte ranges.
pub fn same_finite_coordinates(candidate: &[u8], proposal: &[u8]) -> bool {
    if candidate.len() != proposal.len() || candidate.len() % 4 != 0 {
        return false;
    }
    candidate.chunks_exact(4).zip(proposal.chunks_exact(4)).all(|(a,b)| {
        let lhs = u32::from_ne_bytes([a[0],a[1],a[2],a[3]]);
        let rhs = u32::from_ne_bytes([b[0],b[1],b[2],b[3]]);
        lhs == rhs && f32::from_bits(lhs).is_finite()
    })
}

pub fn reuse_eligible(
    certificate: Option<SweepCertificate>, trial: u32, fraction: f32,
    candidate: &[u8], proposal: &[u8],
) -> bool {
    certificate.is_some_and(SweepCertificate::covers_endpoint)
        && trial == 0 && fraction.to_bits() == 1.0_f32.to_bits()
        && same_finite_coordinates(candidate, proposal)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }
    fn cert() -> Option<SweepCertificate> {
        Some(SweepCertificate { raw_toi: 1.2, ceiling: 1.5, overlapping_start: false })
    }
    #[test]
    fn first_full_trial_identical_finite_proposal_qualifies() {
        let p = bytes(&[1.0,-2.5,0.0]);
        assert!(reuse_eligible(cert(),0,1.0,&p,&p));
    }
    #[test]
    fn no_certificate_and_partial_or_later_trials_fall_back() {
        let p=bytes(&[1.0]);
        assert!(!reuse_eligible(None,0,1.0,&p,&p));
        assert!(!reuse_eligible(cert(),0,0.5,&p,&p));
        assert!(!reuse_eligible(cert(),1,1.0,&p,&p));
    }
    #[test]
    fn signed_zero_nan_infinity_and_truncated_buffers_are_rejected() {
        assert!(!same_finite_coordinates(&bytes(&[0.0]), &bytes(&[-0.0])));
        for n in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let b=bytes(&[n]); assert!(!same_finite_coordinates(&b,&b));
        }
        assert!(!same_finite_coordinates(&[0],&[0]));
        assert!(!same_finite_coordinates(&[0,0,0,0],&[]));
    }
    #[test]
    fn finite_complete_sweep_and_nonoverlap_are_mandatory() {
        let p=bytes(&[1.0]);
        for bad in [
            SweepCertificate {raw_toi:0.99,ceiling:1.5,overlapping_start:false},
            SweepCertificate {raw_toi:1.6,ceiling:1.5,overlapping_start:false},
            SweepCertificate {raw_toi:f32::NAN,ceiling:1.5,overlapping_start:false},
            SweepCertificate {raw_toi:1.0,ceiling:f32::INFINITY,overlapping_start:false},
            SweepCertificate {raw_toi:1.0,ceiling:1.0,overlapping_start:true},
        ] { assert!(!reuse_eligible(Some(bad),0,1.0,&p,&p)); }
    }
    #[test]
    fn cancellation_after_full_fraction_requires_actual_candidate_check() {
        let origin = 16777216.0_f32;
        let proposal = 0.5_f32;
        let actual = origin + (proposal - origin);
        assert_ne!(actual.to_bits(), proposal.to_bits());
        assert!(!reuse_eligible(cert(),0,1.0,&bytes(&[actual]),&bytes(&[proposal])));
    }
}
