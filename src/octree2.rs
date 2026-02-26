use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use supersdf::sdf::*;
use supersdf::vec3::Vec3;

#[derive(Clone)]
pub enum OctreeNode {
    Node {
        center: Vec3,
        size: f32,
        sdf: Rc<DistanceFieldEnum>,
    },
    Empty,
}

/// Return the center offset for child octant `i` given `half` = child_size / 2.
pub fn octant_offset(i: usize, half: f32) -> Vec3 {
    Vec3::new(
        if i & 1 != 0 { half } else { -half },
        if i & 2 != 0 { half } else { -half },
        if i & 4 != 0 { half } else { -half },
    )
}

impl OctreeNode {

    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    pub fn get_child_nodes(&self) -> [OctreeNode; 8] {
        // Subdivide into 8 children
        match self {
            OctreeNode::Node {center, size ,sdf} => {
                let child_size = size / 2.0;
                let children :[OctreeNode;8]= std::array::from_fn(|i| {
                    let child_center = *center + octant_offset(i, child_size / 2.0);
                    OctreeNode::get_node(child_center, child_size, sdf)
                });
            
                children
            }
            OctreeNode::Empty => {
                std::array::from_fn(|i| { OctreeNode::Empty})
            }
        }
    }
    
    pub fn get_node(
        center: Vec3,
        size: f32,
        sdf: &DistanceFieldEnum
    ) -> OctreeNode {
        let optimized = sdf.optimized_for_block(center, size);

        // Empty check
        if optimized.is_empty() {
            return OctreeNode::Empty;
        }

        // AABB vs AABB check — tighter cull than sphere bounds
        let bounds = optimized.calculate_aabb_bounds();
        if bounds.is_finite() && !bounds.overlaps_aabb(center, size / 2.0) {
            return OctreeNode::Empty;
        }

        let half_diag = size * 0.5 * 1.7320508;
        if optimized.distance(center) > half_diag {
            // a fully outside voxel.
            return OctreeNode::Empty;
        }
        
        if optimized.distance(center) < -half_diag {
            // a fully enclosed voxel.
            // it may not be fully, fully enclosed, but at least it looks like it.
            return OctreeNode::Empty;
        }

        OctreeNode::Node {
            center,
            size,
            sdf: optimized
        }
    }


}

/// Build an octree from scratch for an SDF (no old tree to diff against).
pub fn build_octree(sdf: &DistanceFieldEnum, root_size: f32) -> OctreeNode {
    let mut reused_count = 0u32;

    OctreeNode::get_node(
        Vec3::new(0.0, 0.0, 0.0),
        root_size,
        sdf,
    )
}


impl std::hash::Hash for OctreeNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            OctreeNode::Node { center, size, .. } => {
                0u8.hash(state);
                center.x.to_bits().hash(state);
                center.y.to_bits().hash(state);
                center.z.to_bits().hash(state);
                size.to_bits().hash(state);
            }
            OctreeNode::Empty => {
                1u8.hash(state);
            }
        }
    }
}

impl PartialEq for OctreeNode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                OctreeNode::Node { center: c1, size: s1, sdf: sdf1 },
                OctreeNode::Node { center: c2, size: s2, sdf: sdf2 },
                ) => sdf1 == sdf2 &&
                c1.x.to_bits() == c2.x.to_bits()
                && c1.y.to_bits() == c2.y.to_bits()
                && c1.z.to_bits() == c2.z.to_bits()
                && s1.to_bits() == s2.to_bits(),
            
            (OctreeNode::Empty, OctreeNode::Empty) => true,
            _ => false,
        }
    }
}

impl Eq for OctreeNode {}
