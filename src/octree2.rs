use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use supersdf::sdf::*;
use supersdf::vec3::Vec3;

pub const MIN_NODE_SIZE: f32 = 4.0;
pub const ROOT_SIZE: f32 = 10000.0;

#[derive(Clone)]
pub enum OctreeNode {
    Leaf {
        center: Vec3,
        size: f32,
        optimized_sdf: Rc<DistanceFieldEnum>,
    },
    Branch {
        center: Vec3,
        size: f32,
        optimized_sdf: Rc<DistanceFieldEnum>,
        children: [Option<Rc<OctreeNode>>; 8],
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
    
    pub fn build_node(
        center: Vec3,
        size: f32,
        sdf: &DistanceFieldEnum,
        old_node: &OctreeNode,
        cache: &mut HashSet<DistanceFieldEnum>,
        reused_count: &mut u32,
    ) -> OctreeNode {
        let optimized = sdf.optimized_for_block(center, size, cache);

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

        match old_node {
            OctreeNode::Leaf { optimized_sdf, .. }
            | OctreeNode::Branch { optimized_sdf, .. }
                if optimized.equals(optimized_sdf) =>
            {
                // Collect all hashes from the reused subtree so we don't delete their programs
                *reused_count += 1;
                return old_node.clone();
            }
            _ => {}
        }


        // Leaf condition: stop subdividing at min size or <=6 primitives
        if size <= MIN_NODE_SIZE {
            let half = size / 2.0;
            let cell_min = center - Vec3::new(half, half, half);
            let cell_max = center + Vec3::new(half, half, half);
            let content = optimized.calculate_aabb_bounds();
            return OctreeNode::Leaf {
                center,
                size,
                optimized_sdf: optimized
            };
        }

        // Subdivide into 8 children
        let child_size = size / 2.0;
        let children: [Option<Rc<OctreeNode>>; 8] = std::array::from_fn(|i| {
            let child_center = center + octant_offset(i, child_size / 2.0);
            let old_child = match old_node {
                OctreeNode::Branch { children, .. } => children[i]
                    .as_ref()
                    .map(|rc| rc.as_ref())
                    .unwrap_or(&OctreeNode::Empty),
                _ => &OctreeNode::Empty,
            };
            let child = Self::build_node(
                child_center,
                child_size,
                &optimized,
                old_child,
                cache,
                reused_count,
            );

            match child {
                OctreeNode::Empty => None,
                node => Some(Rc::new(node)),
            }
        });

        // Count non-empty children
        let non_empty_count = children.iter().filter(|c| c.is_some()).count();

        // All children empty → this branch is empty
        if non_empty_count == 0 {
            return OctreeNode::Empty;
        }

        OctreeNode::Branch {
            center,
            size,
            optimized_sdf: optimized,
            children,
        }
    }


    pub fn count_leaves(&self) -> u32 {
        match self {
            OctreeNode::Empty => 0,
            OctreeNode::Leaf { .. } => 1,
            OctreeNode::Branch { children, .. } => {
                children.iter().flatten().map(|c| c.count_leaves()).sum()
            }
        }
    }

    pub fn for_each_leaf<F>(&self, f: &mut F)
    where
        F: FnMut(&OctreeNode),
    {
        match self {
            OctreeNode::Empty => {}
            OctreeNode::Leaf { .. } => f(self),
            OctreeNode::Branch { children, .. } => {
                for child in children.iter().flatten() {
                    child.for_each_leaf(f);
                }
            }
        }
    }

    pub fn count_branches(node: &OctreeNode) -> u32 {
        match node {
            OctreeNode::Empty => 0,
            OctreeNode::Leaf { .. } => 0,
            OctreeNode::Branch { children, .. } => {
                1 + children.iter().flatten().map(|c| Self::count_branches(c)).sum::<u32>()
            }
        }
    }
}

/// Build an octree from scratch for an SDF (no old tree to diff against).
pub fn build_octree(sdf: &DistanceFieldEnum, root_size: f32) -> OctreeNode {
    let mut cache = HashSet::new();
    let mut reused_count = 0u32;

    OctreeNode::build_node(
        Vec3::new(0.0, 0.0, 0.0),
        root_size,
        sdf,
        &OctreeNode::Empty,
        &mut cache,
        &mut reused_count,
    )
}
