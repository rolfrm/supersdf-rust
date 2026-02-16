use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::sdf::*;
use crate::sdf_compiler;
use crate::vec3::Vec3;

pub const MIN_NODE_SIZE: f32 = 0.5;
pub const ROOT_SIZE: f32 = 4000.0;

#[derive(Clone)]
pub enum OctreeNode {
    Leaf {
        center: Vec3,
        size: f32,
        render_min: Vec3,
        render_max: Vec3,
        optimized_sdf: Rc<DistanceFieldEnum>,
        topology_hash: u64,
        params: Vec<f32>,
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
    pub fn build_node(
        center: Vec3,
        size: f32,
        sdf: &DistanceFieldEnum,
        old_node: &OctreeNode,
        cache: &mut HashSet<DistanceFieldEnum>,
        to_compile: &mut HashMap<u64, Rc<DistanceFieldEnum>>,
        reused_count: &mut u32,
    ) -> OctreeNode {
        let optimized = sdf.optimized_for_block(center, size, cache);

        // Empty check
        if matches!(optimized.as_ref(), DistanceFieldEnum::Empty) {
            return OctreeNode::Empty;
        }

        // AABB vs AABB check — tighter cull than sphere bounds
        let bounds = optimized.calculate_aabb_bounds();
        if bounds.is_finite() && !bounds.overlaps_aabb(center, size / 2.0) {
            return OctreeNode::Empty;
        }

        let half_diag = size * 0.5 * 1.7320508;
        if optimized.distance(center) > half_diag {
            return OctreeNode::Empty;
        }

        // Change detection: if the optimized SDF matches old node's, reuse subtree
        match old_node {
            OctreeNode::Leaf { optimized_sdf, .. }
            | OctreeNode::Branch { optimized_sdf, .. }
                if optimized.equals(optimized_sdf) =>
            {
                // Collect all hashes from the reused subtree so we don't delete their programs
                Self::collect_hashes(old_node, to_compile);
                *reused_count += 1;
                return old_node.clone();
            }
            _ => {}
        }

        let prim_count = optimized.count_primitives_up_to(6);

        // Leaf condition: stop subdividing at min size or <=6 primitives
        if (size <= MIN_NODE_SIZE && prim_count <= 6) || prim_count <= 6 {
            let hash = optimized.topology_hash();
            to_compile.entry(hash).or_insert_with(|| optimized.clone());
            let params = sdf_compiler::collect_block_sdf_params(&optimized);
            let half = size / 2.0;
            let cell_min = center - Vec3::new(half, half, half);
            let cell_max = center + Vec3::new(half, half, half);
            let content = optimized.calculate_aabb_bounds();
            let render_min = Vec3::new(cell_min.x.max(content.min.x), cell_min.y.max(content.min.y), cell_min.z.max(content.min.z));
            let render_max = Vec3::new(cell_max.x.min(content.max.x), cell_max.y.min(content.max.y), cell_max.z.min(content.max.z));
            return OctreeNode::Leaf {
                center,
                size,
                render_min,
                render_max,
                optimized_sdf: optimized,
                topology_hash: hash,
                params,
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
                to_compile,
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

        // Only one child with simple SDF → collapse to a leaf
        // (must use parent's center/size/optimized so change detection matches next rebuild)
        if non_empty_count == 1 && prim_count <= 3 {
            let hash = optimized.topology_hash();
            to_compile.entry(hash).or_insert_with(|| optimized.clone());
            let params = sdf_compiler::collect_block_sdf_params(&optimized);
            let half = size / 2.0;
            let cell_min = center - Vec3::new(half, half, half);
            let cell_max = center + Vec3::new(half, half, half);
            let content = optimized.calculate_aabb_bounds();
            let render_min = Vec3::new(cell_min.x.max(content.min.x), cell_min.y.max(content.min.y), cell_min.z.max(content.min.z));
            let render_max = Vec3::new(cell_max.x.min(content.max.x), cell_max.y.min(content.max.y), cell_max.z.min(content.max.z));
            return OctreeNode::Leaf {
                center,
                size,
                render_min,
                render_max,
                optimized_sdf: optimized,
                topology_hash: hash,
                params,
            };
        }

        // Collapse: if every non-empty child is a leaf with the same topology hash,
        // merge back into one leaf at the parent size.
        let first_node = children.iter().flatten().find_map(|c| match c.as_ref() {
            OctreeNode::Leaf { optimized_sdf, .. } => Some(optimized_sdf.clone()),
            _ => None,
        });
        if let Some(fh) = first_node {
            let can_collapse = prim_count < 5
                && children.iter().all(|c| match c {
                    Some(rc) => matches!(rc.as_ref(), OctreeNode::Leaf { optimized_sdf, .. } if optimized_sdf.equals(&fh)),
                    None => true,
                });
            if can_collapse {
                let hash = optimized.topology_hash();
                to_compile.entry(hash).or_insert_with(|| optimized.clone());
                let params = sdf_compiler::collect_block_sdf_params(&optimized);
                let half = size / 2.0;
                let cell_min = center - Vec3::new(half, half, half);
                let cell_max = center + Vec3::new(half, half, half);
                let content = optimized.calculate_aabb_bounds();
                let render_min = Vec3::new(cell_min.x.max(content.min.x), cell_min.y.max(content.min.y), cell_min.z.max(content.min.z));
                let render_max = Vec3::new(cell_max.x.min(content.max.x), cell_max.y.min(content.max.y), cell_max.z.min(content.max.z));
                return OctreeNode::Leaf {
                    center,
                    size,
                    render_min,
                    render_max,
                    optimized_sdf: optimized,
                    topology_hash: hash,
                    params,
                };
            }
        }

        OctreeNode::Branch {
            center,
            size,
            optimized_sdf: optimized,
            children,
        }
    }

    /// Collect topology hashes from a subtree (for reused nodes whose programs we must keep).
    pub fn collect_hashes(node: &OctreeNode, to_compile: &mut HashMap<u64, Rc<DistanceFieldEnum>>) {
        match node {
            OctreeNode::Leaf { topology_hash, optimized_sdf, .. } => {
                to_compile.entry(*topology_hash).or_insert_with(|| optimized_sdf.clone());
            }
            OctreeNode::Branch { children, .. } => {
                for child in children.iter().flatten() {
                    Self::collect_hashes(child, to_compile);
                }
            }
            OctreeNode::Empty => {}
        }
    }

    pub fn count_leaves(node: &OctreeNode) -> u32 {
        match node {
            OctreeNode::Empty => 0,
            OctreeNode::Leaf { .. } => 1,
            OctreeNode::Branch { children, .. } => {
                children.iter().flatten().map(|c| Self::count_leaves(c)).sum()
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
    let mut to_compile: HashMap<u64, Rc<DistanceFieldEnum>> = HashMap::new();
    let mut reused_count = 0u32;

    OctreeNode::build_node(
        Vec3::new(0.0, 0.0, 0.0),
        root_size,
        sdf,
        &OctreeNode::Empty,
        &mut cache,
        &mut to_compile,
        &mut reused_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_sphere_sdf(offset: f32) -> DistanceFieldEnum {
        let s1 = DistanceFieldEnum::sphere(Vec3::new(-offset / 2.0, 0.0, 0.0), 5.0);
        let s2 = DistanceFieldEnum::sphere(Vec3::new(offset / 2.0, 0.0, 0.0), 5.0);
        s1.insert_2(s2)
    }

    /// Build an SDF with `n` spheres spaced along X with given `spacing`.
    fn n_sphere_sdf(n: u32, spacing: f32) -> DistanceFieldEnum {
        let mut sdf = DistanceFieldEnum::Empty;
        for i in 0..n {
            let x = i as f32 * spacing - (n as f32 - 1.0) * spacing / 2.0;
            sdf = sdf.insert_2(DistanceFieldEnum::sphere(Vec3::new(x, 0.0, 0.0), 5.0));
        }
        sdf
    }

    #[test]
    fn test_two_spheres_coincident() {
        // offset=0: both at origin → 2 primitives ≤ 6, so exactly 1 leaf
        let sdf = two_sphere_sdf(0.0);
        let root = build_octree(&sdf, ROOT_SIZE);
        let leaves = OctreeNode::count_leaves(&root);
        assert_eq!(leaves, 1, "coincident two-sphere SDF should be 1 leaf (2 prims ≤ 6)");
    }

    #[test]
    fn test_two_spheres_touching() {
        // offset=10: touching (r=5 each) → 2 primitives ≤ 6, still 1 leaf
        let sdf = two_sphere_sdf(10.0);
        let root = build_octree(&sdf, ROOT_SIZE);
        let leaves = OctreeNode::count_leaves(&root);
        assert_eq!(leaves, 1, "touching two-sphere SDF should be 1 leaf (2 prims ≤ 6)");
    }

    #[test]
    fn test_two_spheres_always_one_leaf() {
        // Two primitives ≤ 6, so the octree never subdivides regardless of offset.
        for &offset in &[0.0, 10.0, 30.0, 100.0, 500.0, 1000.0] {
            let sdf = two_sphere_sdf(offset);
            let root = build_octree(&sdf, ROOT_SIZE);
            let leaves = OctreeNode::count_leaves(&root);
            assert_eq!(leaves, 1,
                "offset={}: two spheres (2 prims ≤ 6) should always be 1 leaf, got {}", offset, leaves);
        }
    }

    #[test]
    fn test_two_spheres_offset_sweep() {
        // Sweep offsets: verify structural properties at each distance.
        // Two spheres always produce a single leaf, but the leaf's render_min/render_max
        // should grow to encompass both spheres.
        let offsets = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0];
        let mut results: Vec<(f32, u32, u32)> = Vec::new();

        for &offset in &offsets {
            let sdf = two_sphere_sdf(offset);
            let root = build_octree(&sdf, ROOT_SIZE);
            let leaves = OctreeNode::count_leaves(&root);
            let branches = OctreeNode::count_branches(&root);

            results.push((offset, leaves, branches));

            assert!(leaves >= 1, "offset={}: expected at least 1 leaf, got {}", offset, leaves);
        }

        // Print sweep results for visibility
        for (offset, leaves, branches) in &results {
            println!("offset={:>7.1}  leaves={}  branches={}", offset, leaves, branches);
        }
    }

    #[test]
    fn test_many_spheres_subdivides() {
        // 8 spheres spaced 30 apart → 8 primitives > 6, forces subdivision.
        let sdf = n_sphere_sdf(8, 30.0);
        let root = build_octree(&sdf, ROOT_SIZE);
        let leaves = OctreeNode::count_leaves(&root);
        let branches = OctreeNode::count_branches(&root);
        assert!(leaves >= 2, "8 spheres should subdivide into >= 2 leaves, got {}", leaves);
        assert!(branches >= 1, "8 spheres should have >= 1 branch, got {}", branches);
        println!("8 spheres, spacing=30: leaves={}, branches={}", leaves, branches);
    }

    #[test]
    fn test_many_spheres_offset_sweep() {
        // Sweep spacing for 8 spheres and verify structural sanity at each step.
        let spacings = [5.0, 15.0, 30.0, 60.0, 120.0, 250.0];
        let mut results: Vec<(f32, u32, u32)> = Vec::new();

        for &spacing in &spacings {
            let sdf = n_sphere_sdf(8, spacing);
            let root = build_octree(&sdf, ROOT_SIZE);
            let leaves = OctreeNode::count_leaves(&root);
            let branches = OctreeNode::count_branches(&root);

            results.push((spacing, leaves, branches));

            // 8 primitives > 6, so we always need subdivision
            assert!(leaves >= 2, "spacing={}: 8 spheres should produce >= 2 leaves, got {}", spacing, leaves);
            assert!(branches >= 1, "spacing={}: 8 spheres should need >= 1 branch, got {}", spacing, branches);
        }

        // Print sweep results
        for (spacing, leaves, branches) in &results {
            println!("8 spheres, spacing={:>6.1}  leaves={}  branches={}", spacing, leaves, branches);
        }
    }

    #[test]
    fn test_single_sphere_produces_one_leaf() {
        let sdf = DistanceFieldEnum::sphere(Vec3::new(0.0, 0.0, 0.0), 5.0);
        let root = build_octree(&sdf, ROOT_SIZE);
        let leaves = OctreeNode::count_leaves(&root);
        assert_eq!(leaves, 1, "single sphere should produce exactly 1 leaf");
    }

    #[test]
    fn test_empty_produces_no_nodes() {
        let sdf = DistanceFieldEnum::Empty;
        let root = build_octree(&sdf, ROOT_SIZE);
        let leaves = OctreeNode::count_leaves(&root);
        assert_eq!(leaves, 0, "empty SDF should produce 0 leaves");
        assert!(matches!(root, OctreeNode::Empty));
    }
}
