//! In-memory effective continuous selection derived from validated joints only.
//! Does not alter sealed wire records, output, or device control POD.
use std::collections::HashSet;
use crate::{joint::OwnedJointScene, joint_angular_control_gpu::COORDINATE_MODEL};

pub const CONTINUOUS_MODEL: &str = "parent_local_swing_twist_x_continuous_v2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwistOverlay {
    pub scene_uuid: String,
    pub joint_uuids: Vec<String>,
    /// Empty means this joint has no directional control.
    pub source_models: Vec<String>,
    pub selected_joint_uuids: Vec<String>,
    pub mask: Vec<bool>,
}

/// Select only genuine validated two-PDRD-endpoint directional records.
/// Unselected isotropic/FEM rows and original model strings remain unchanged.
pub fn from_validated(scene: &OwnedJointScene) -> Result<TwistOverlay, &'static str> {
    select_model(scene, COORDINATE_MODEL)
}

pub fn fresh_from_validated(scene: &OwnedJointScene) -> Result<TwistOverlay, &'static str> {
    select_model(scene, CONTINUOUS_MODEL)
}

/// Independent v3 selection. A world parent is a constant frame, not a body.
pub fn material_spin_from_validated(scene:&OwnedJointScene)->Result<TwistOverlay,&'static str>{
    let bodies:HashSet<u32>=scene.bodies.values().copied().collect();
    let mut out=TwistOverlay{scene_uuid:scene.scene_uuid.clone(),joint_uuids:Vec::new(),source_models:Vec::new(),selected_joint_uuids:Vec::new(),mask:Vec::new()};
    for joint in &scene.joints {
        let model=joint.wire.angular_control_v1.as_ref().map(|c|c.coordinate_model.as_str()).unwrap_or("");
        let a=joint.parent_runtime_body_index;let b=joint.child_runtime_body_index;
        let selected=joint.wire.enabled && model==crate::material_spin_history::MODEL && (a==u32::MAX||bodies.contains(&a)) && (b==u32::MAX||bodies.contains(&b)) && (a!=u32::MAX||b!=u32::MAX);
        out.joint_uuids.push(joint.wire.joint_uuid.clone());out.source_models.push(model.to_owned());out.mask.push(selected);
        if selected{out.selected_joint_uuids.push(joint.wire.joint_uuid.clone());}
    }
    if out.selected_joint_uuids.is_empty(){return Err("MATERIAL_SPIN_NO_ELIGIBLE_JOINTS");}Ok(out)
}

/// FEM tree rows are the sealed prefix; appended cross-links have no angular
/// material state and remain false in the full device-row selection.
pub(crate) fn fem_material_spin_from_owned(scene:&crate::soft_joint::OwnedSoftScene)->Result<Option<TwistOverlay>,&'static str>{
    if scene.records.len()<scene.wires.len(){return Err("FEM_MATERIAL_SPIN_INVALID: missing tree records");}
    let mut out=TwistOverlay{scene_uuid:scene.scene_uuid.clone(),joint_uuids:Vec::new(),source_models:Vec::new(),selected_joint_uuids:Vec::new(),mask:vec![false;scene.records.len()]};
    for (i,row) in scene.wires.iter().enumerate(){
        let model=row.wire.angular_control_v1.as_ref().map(|c|c.coordinate_model.as_str()).unwrap_or("");
        let selected=row.wire.enabled && model==crate::material_spin_history::MODEL;
        if selected && (scene.records[i].enabled!=1 || scene.records[i].control.enabled!=1 || scene.records[i].control.coordinate_model!=1){return Err("FEM_MATERIAL_SPIN_INVALID: selected row lacks directional control");}
        out.joint_uuids.push(row.wire.joint_uuid.clone());out.source_models.push(model.to_owned());out.mask[i]=selected;
        if selected {out.selected_joint_uuids.push(row.wire.joint_uuid.clone());}
    }
    if out.selected_joint_uuids.is_empty(){Ok(None)}else{Ok(Some(out))}
}

fn select_model(scene: &OwnedJointScene, expected: &str) -> Result<TwistOverlay, &'static str> {
    let body_indices: HashSet<u32> = scene.bodies.values().copied().collect();
    let mut out = TwistOverlay { scene_uuid:scene.scene_uuid.clone(), joint_uuids:Vec::new(),
        source_models:Vec::new(),selected_joint_uuids:Vec::new(),mask:Vec::new() };
    for joint in &scene.joints {
        let model = joint.wire.angular_control_v1.as_ref().map(|c|c.coordinate_model.as_str()).unwrap_or("");
        let selected = eligible(joint.wire.enabled, model, expected, joint.parent_runtime_body_index, joint.child_runtime_body_index, &body_indices);
        out.joint_uuids.push(joint.wire.joint_uuid.clone());
        out.source_models.push(model.to_owned());
        out.mask.push(selected);
        if selected { out.selected_joint_uuids.push(joint.wire.joint_uuid.clone()); }
    }
    if out.selected_joint_uuids.is_empty() { return Err("TWIST_RESUME_NO_ELIGIBLE_JOINTS") }
    Ok(out)
}

impl TwistOverlay {
    /// Assemble backend context solely from the validated scene overlay. The
    /// sidecar is never an authority for scene, models, IDs or selection.
    pub(crate) fn checkpoint_context(self, explicit_migration: bool) -> crate::backend::TwistCheckpointContext {
        let mode = crate::checkpoint_history::TwistModeMetadata {
            fit_scheme: "selected_covariance_polar_svd_v1".into(),
            effective_model: CONTINUOUS_MODEL.into(),
            source_models: self.source_models.clone(),
            selected_joint_uuids: self.selected_joint_uuids.clone(),
            migration_time: None,
        };
        crate::backend::TwistCheckpointContext { scene:self.scene_uuid, joints:self.joint_uuids,
            sealed_models:self.source_models, selected:self.selected_joint_uuids,
            mask:self.mask, mode, explicit_migration }
    }
}

fn eligible(enabled: bool, model: &str, expected: &str, parent: u32, child: u32, bodies: &HashSet<u32>) -> bool {
    enabled && model == expected && bodies.contains(&parent) && bodies.contains(&child)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selects_only_validated_directional_two_pdrd_endpoints() {
        let bodies = HashSet::from([0, 1]);
        assert!(eligible(true, COORDINATE_MODEL, COORDINATE_MODEL,0,1,&bodies));
        assert!(!eligible(false, COORDINATE_MODEL, COORDINATE_MODEL,0,1,&bodies));
        assert!(!eligible(true, COORDINATE_MODEL, COORDINATE_MODEL,u32::MAX,1,&bodies));
        assert!(!eligible(true, COORDINATE_MODEL, COORDINATE_MODEL,0,2,&bodies));
        assert!(!eligible(true, "", COORDINATE_MODEL,0,1,&bodies));
        assert!(!eligible(true, "isotropic", COORDINATE_MODEL,0,1,&bodies));
    }
}
