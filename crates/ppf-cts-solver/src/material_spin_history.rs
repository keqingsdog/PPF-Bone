//! Transactional material-spin history. V2 twist history is intentionally separate.
//!
//! The owner supplies a certified candidate from the actual accepted-to-trial
//! path. This container does not infer a path from endpoint quaternions.
pub(crate) const MODEL: &str = "material_spin_path_v3";

/// Device ABI shared with material-spin kernel; a zero model is a true no-op.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MaterialSpinContext {
    pub accepted_q_wxyz: [f32; 4],
    pub accepted_a: [f32; 3],
    pub model: u32,
}
unsafe impl ppf_cts_compute::Pod for MaterialSpinContext {}
const _: () = assert!(std::mem::size_of::<MaterialSpinContext>() == 32);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct MaterialSpinRow {
    /// Joint-relative accepted quaternion, scalar first.
    pub q_wxyz: [f64; 4],
    pub a: [f64; 3],
}

impl MaterialSpinRow {
    fn valid(&self) -> bool {
        let norm2: f64 = self.q_wxyz.iter().map(|x| x*x).sum();
        self.a.iter().chain(self.q_wxyz.iter()).all(|x| x.is_finite())
            && norm2.is_finite() && (norm2 - 1.0).abs() <= 1e-8
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct AcceptedMaterialSpin {
    pub scene_uuid: String,
    pub joint_uuids: Vec<String>,
    pub time: f64,
    pub generation: u64,
    /// Selected joints in sealed full-joint order. No V2 theta-only migration.
    pub rows: Vec<MaterialSpinRow>,
}

impl AcceptedMaterialSpin {
    pub fn validate(&self, scene: &str, joints: &[String], time: f64) -> Result<(), &'static str> {
        if self.scene_uuid != scene || self.joint_uuids != joints || self.rows.len() != joints.len()
            || joints.is_empty() || !time.is_finite() || time < 0.0 || self.time.to_bits() != time.to_bits()
            || self.rows.iter().any(|row| !row.valid()) {
            return Err("material spin history identity, time or row invalid");
        }
        Ok(())
    }

    /// Create a candidate without mutating the accepted generation.
    pub fn trial(&self) -> TrialMaterialSpin {
        TrialMaterialSpin { base_generation:self.generation, rows: self.rows.clone() }
    }

    /// Prepare a new generation; caller publishes it only at the C5 boundary.
    pub fn prepare_commit(&self, trial: TrialMaterialSpin, time: f64) -> Result<Self, &'static str> {
        if !time.is_finite() || time <= self.time || trial.base_generation != self.generation || trial.rows.len() != self.rows.len()
            || trial.rows.iter().any(|row| !row.valid()) {
            return Err("material spin commit invalid");
        }
        let generation=self.generation.checked_add(1).ok_or("material spin generation overflow")?;
        Ok(Self { scene_uuid: self.scene_uuid.clone(), joint_uuids: self.joint_uuids.clone(), time, generation, rows: trial.rows })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TrialMaterialSpin {
    base_generation: u64,
    pub rows: Vec<MaterialSpinRow>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trial_rollback_and_single_generation_prepare() {
        let original=AcceptedMaterialSpin { scene_uuid:"s".into(), joint_uuids:vec!["j".into()],
            time:0., generation:0, rows:vec![MaterialSpinRow {q_wxyz:[1.,0.,0.,0.],a:[0.,0.,0.]}] };
        assert!(original.validate("s", &["j".into()],0.).is_ok());
        let mut trial=original.trial();
        trial.rows[0].a=[12.566370614359172,3.490658503988659,0.];
        let next=original.prepare_commit(trial,0.1).unwrap();
        assert_eq!(original.rows[0].a,[0.;3]);
        assert_eq!(next.rows[0].a[1],3.490658503988659);
        assert!(next.prepare_commit(next.trial(),0.1).is_err());
        let mut invalid=next.trial();invalid.rows[0].q_wxyz=[0.;4];
        assert!(next.prepare_commit(invalid,0.2).is_err());
        assert!(next.validate("other", &["j".into()],0.1).is_err());
    }
}
