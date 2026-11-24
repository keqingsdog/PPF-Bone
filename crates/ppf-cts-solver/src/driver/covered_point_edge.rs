//! Proof for the standalone point-edge CCD visitor's empty domain.
use std::sync::OnceLock;

pub(crate) fn requested() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PPF_CCD_SKIP_COVERED_POINT_EDGE").as_deref() == Ok("1"))
}

pub(crate) fn trace() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PPF_CCD_SKIP_COVERED_POINT_EDGE_TRACE").as_deref() == Ok("1")
    })
}

/// Certify the CSR that has just been uploaded for this exact topology.
/// Empty or inconsistent rows keep the complete standalone traversal.
pub(crate) fn covers_every_edge(
    edges: &[u32],
    faces: &[u32],
    vertices: usize,
    indices: &[u32],
    offsets: &[u32],
) -> bool {
    let rows = edges.len() / 2;
    if rows == 0
        || edges.len() % 2 != 0
        || faces.len() % 3 != 0
        || offsets.len() != rows + 1
        || offsets.first() != Some(&0)
        || offsets.last().map(|&n| n as usize) != Some(indices.len())
        || offsets.windows(2).any(|bounds| bounds[0] >= bounds[1])
    {
        return false;
    }

    for (edge, bounds) in edges.chunks_exact(2).zip(offsets.windows(2)) {
        let (a, b) = (edge[0], edge[1]);
        if a == b || a as usize >= vertices || b as usize >= vertices {
            return false;
        }
        for &face in &indices[bounds[0] as usize..bounds[1] as usize] {
            let face = face as usize;
            if face >= faces.len() / 3 {
                return false;
            }
            let triangle = &faces[3 * face..3 * face + 3];
            if !triangle.contains(&a) || !triangle.contains(&b)
                || triangle.iter().any(|&v| v as usize >= vertices)
            {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const EDGES: &[u32] = &[0, 1, 1, 2, 2, 0];
    const FACES: &[u32] = &[0, 1, 2];

    #[test]
    fn open_boundary_edges_each_with_one_face_are_covered() {
        assert!(covers_every_edge(EDGES, FACES, 3, &[0, 0, 0], &[0, 1, 2, 3]));
    }

    #[test]
    fn isolated_point_does_not_invalidate_face_covered_edges() {
        assert!(covers_every_edge(EDGES, FACES, 4, &[0, 0, 0], &[0, 1, 2, 3]));
    }

    #[test]
    fn missing_empty_or_partial_adjacency_keeps_full_traversal() {
        assert!(!covers_every_edge(&[], &[], 3, &[], &[0]));
        assert!(!covers_every_edge(EDGES, FACES, 3, &[], &[]));
        assert!(!covers_every_edge(&[0, 1, 1, 3], FACES, 4, &[0], &[0, 1, 1]));
        assert!(!covers_every_edge(&[0, 1], &[], 2, &[], &[0, 0]));
    }

    #[test]
    fn malformed_csr_or_unrelated_face_keeps_full_traversal() {
        assert!(!covers_every_edge(EDGES, FACES, 3, &[0, 0], &[0, 1, 2, 3]));
        assert!(!covers_every_edge(EDGES, FACES, 3, &[0, 0, 0], &[0, 2, 1, 3]));
        assert!(!covers_every_edge(EDGES, FACES, 3, &[0, 1, 0], &[0, 1, 2, 3]));
        assert!(!covers_every_edge(&[0, 3], FACES, 4, &[0], &[0, 1]));
        assert!(!covers_every_edge(&[0, 1], &[0, 1, 3], 3, &[0], &[0, 1]));
    }
}
