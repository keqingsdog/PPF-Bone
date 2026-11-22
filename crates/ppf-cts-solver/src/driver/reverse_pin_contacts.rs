//! Classify the few final FE records that could have entered through an
//! explicitly permitted hard-pin edge. The device scan remains the authority
//! for batch discovery and for the count beyond its retained records.

use ppf_cts_compute::{Buffer, Device, ReadbackBuffer};

use crate::data::{
    DataSet, FixPair, IntersectionRecord, PullPair, VertexProp, NO_OBJECT_INDEX,
};
use super::intersection::Report;
use super::reverse_pin_fe::authorized_transition;
use super::scene::{FatalResult, SceneView};

#[derive(PartialEq, Eq)]
pub(super) struct Pair {
    face: u32,
    edge: u32,
    // Roles are part of the identity: a different free corner changes the
    // edge that carries the pin permission.
    vertices: [u32; 5],
}

impl Pair {
    fn matches(&self, record: &IntersectionRecord) -> bool {
        record.itype == 0 && self.face == record.elem0 && self.edge == record.elem1
    }
}

fn permitted_hard(prop: &VertexProp) -> bool {
    !prop.collider && prop.fix_index != 0 && prop.pull_index == 0
        && prop.pin_allow_intersection && prop.pdrd_body_index == 0 && prop.volume == 0.0
}

fn ordinary_free(prop: &VertexProp) -> bool {
    !prop.collider && prop.fix_index == 0 && prop.pull_index == 0
        && !prop.pin_allow_intersection && prop.pdrd_body_index == 0
        && prop.volume == 0.0 && prop.mass > 0.0
}

fn active_soft(prop: &VertexProp, vertex: u32, pull: &[PullPair]) -> bool {
    prop.collider && prop.fix_index == 0 && !prop.pin_allow_intersection
        && prop.pdrd_body_index == 0 && prop.pull_index.checked_sub(1)
            .and_then(|i| pull.get(i as usize))
            .is_some_and(|pin| pin.index == vertex)
}

unsafe fn matching_pair(
    view: &SceneView,
    rod_count: u32,
    props: &[VertexProp],
    pull: &[PullPair],
    record: &IntersectionRecord,
) -> Option<Pair> {
    if record.itype != 0 || record.elem1 < rod_count {
        return None;
    }
    let face = view.faces().get(record.elem0 as usize)?;
    let edge = view.edges().get(record.elem1 as usize)?;
    if view.face_props_mut().get(record.elem0 as usize)?.collider {
        return None;
    }
    let p0 = props.get(face[0] as usize)?;
    let p1 = props.get(face[1] as usize)?;
    let p2 = props.get(face[2] as usize)?;
    let q0 = props.get(edge[0] as usize)?;
    let q1 = props.get(edge[1] as usize)?;
    let cloth_owner = p0.object_index;
    let sleeve_owner = q0.object_index;
    if cloth_owner == NO_OBJECT_INDEX || sleeve_owner == NO_OBJECT_INDEX
        || cloth_owner == sleeve_owner || p1.object_index != cloth_owner
        || p2.object_index != cloth_owner || q1.object_index != sleeve_owner
        || !active_soft(q0, edge[0], pull) || !active_soft(q1, edge[1], pull)
    {
        return None;
    }
    let vertices = match (ordinary_free(p0), ordinary_free(p1), ordinary_free(p2)) {
        (true, false, false) if permitted_hard(p1) && permitted_hard(p2) =>
            [face[1], face[2], face[0], edge[0], edge[1]],
        (false, true, false) if permitted_hard(p2) && permitted_hard(p0) =>
            [face[2], face[0], face[1], edge[0], edge[1]],
        (false, false, true) if permitted_hard(p0) && permitted_hard(p1) =>
            [face[0], face[1], face[2], edge[0], edge[1]],
        _ => return None,
    };
    Some(Pair { face: record.elem0, edge: record.elem1, vertices })
}

fn points(positions: &[f32], vertices: [u32; 5]) -> Option<[[f32; 3]; 5]> {
    let mut result = [[0.0; 3]; 5];
    for (point, vertex) in result.iter_mut().zip(vertices) {
        let first = (vertex as usize).checked_mul(3)?;
        point.copy_from_slice(positions.get(first..first.checked_add(3)?)?);
    }
    Some(result)
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn filter_trial<D: Device>(
    device: &mut D,
    data: &DataSet,
    accepted: &mut ReadbackBuffer<f32>,
    candidate: &[f32],
    props: &[VertexProp],
    fix: &Buffer<FixPair>,
    pull: &[PullPair],
    previous: &[Pair],
    report: &mut Report,
) -> FatalResult<Vec<Pair>> {
    if report.is_clean() || fix.len() < 2 || pull.len() < 2 {
        return Ok(Vec::new());
    }

    let view = SceneView::new(data);
    let mut admitted = Vec::new();
    let mut have_accepted_positions = false;
    for record in &report.records {
        let Some(pair) = matching_pair(&view, data.rod_count, props, pull, record) else {
            continue;
        };
        let [hard0, hard1, ..] = pair.vertices;
        let hard0_slot = props[hard0 as usize].fix_index as usize - 1;
        let hard1_slot = props[hard1 as usize].fix_index as usize - 1;
        if hard0_slot >= fix.len() || hard1_slot >= fix.len() {
            continue;
        }
        // VertexProp carries the vertex's authored allowance. Require the
        // actual active FIX to carry it too, including after a pin switch.
        let pin0 = fix.read_one(device, hard0_slot)?;
        let pin1 = fix.read_one(device, hard1_slot)?;
        if pin0.index != hard0 || pin1.index != hard1
            || !pin0.allow_intersection || !pin1.allow_intersection
        {
            continue;
        }
        if !have_accepted_positions {
            accepted.download(device)?;
            have_accepted_positions = true;
        }
        let (Some(start), Some(end)) = (
            points(accepted.host(), pair.vertices), points(candidate, pair.vertices),
        ) else {
            continue;
        };
        if authorized_transition(start, end, previous.contains(&pair)) {
            admitted.push(pair);
        }
    }
    report.records.retain(|record| !admitted.iter().any(|pair| pair.matches(record)));
    report.found -= admitted.len() as u32;
    Ok(admitted)
}
