use std::rc::Rc;

use supersdf::sdf::*;
use supersdf::vec3::Vec3;

#[derive(Clone)]
pub enum OctreeNode {
    Node {
        center: Vec3,
        size: f32,
        sdf: Rc<Sdf>,
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
                std::array::from_fn(|_i| { OctreeNode::Empty})
            }
        }
    }
    
    pub fn get_node(
        center: Vec3,
        size: f32,
        sdf: &Sdf
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

/// Look up the node containing `point` at the given `target_size`.
/// Walks from `root` down through the octree, using `child_cache` to
/// avoid recomputing children, until the node size matches `target_size`.
pub fn lookup_node(
    root: &OctreeNode,
    child_cache: &mut std::collections::HashMap<OctreeNode, [OctreeNode; 8]>,
    point: Vec3,
    target_size: f32,
) -> OctreeNode {
    let mut current = root.clone();

    loop {
        match &current {
            OctreeNode::Empty => return OctreeNode::Empty,
            OctreeNode::Node { center, size, .. } => {
                if *size <= target_size {
                    return current;
                }

                let children = child_cache
                    .entry(current.clone())
                    .or_insert_with(|| current.get_child_nodes());

                // Determine which octant the point falls in
                let octant = ((point.x >= center.x) as usize)
                    | (((point.y >= center.y) as usize) << 1)
                    | (((point.z >= center.z) as usize) << 2);

                current = children[octant].clone();
            }
        }
    }
}

/// Recursively remove a node and all its cached descendants from the cache.
fn invalidate_cache(
    node: &OctreeNode,
    cache: &mut std::collections::HashMap<OctreeNode, [OctreeNode; 8]>,
) {
    if let Some(children) = cache.remove(node) {
        for child in &children {
            invalidate_cache(child, cache);
        }
    }
}

/// Edit all nodes at `target_size` whose AABBs overlap the edit region.
/// `edit_center` and `edit_half` define the AABB of the edit region.
/// `edit_fn` is applied to each overlapping node's SDF.
pub fn edit_node(
    root: &OctreeNode,
    child_cache: &mut std::collections::HashMap<OctreeNode, [OctreeNode; 8]>,
    edit_center: Vec3,
    edit_half: Vec3,
    target_size: f32,
    edit_fn: &dyn Fn(Rc<Sdf>) -> Rc<Sdf>,
) {
    edit_node_recursive(root, child_cache, edit_center, edit_half, target_size, edit_fn, None);
}

fn aabb_overlap(a_center: Vec3, a_half: f32, b_center: Vec3, b_half: Vec3) -> bool {
    (a_center.x - b_center.x).abs() <= a_half + b_half.x
        && (a_center.y - b_center.y).abs() <= a_half + b_half.y
        && (a_center.z - b_center.z).abs() <= a_half + b_half.z
}

fn edit_node_recursive(
    node: &OctreeNode,
    child_cache: &mut std::collections::HashMap<OctreeNode, [OctreeNode; 8]>,
    edit_center: Vec3,
    edit_half: Vec3,
    target_size: f32,
    edit_fn: &dyn Fn(Rc<Sdf>) -> Rc<Sdf>,
    parent_and_octant: Option<(&OctreeNode, usize)>,
) {
    match node {
        OctreeNode::Empty => return,
        OctreeNode::Node { center, size, sdf } => {
            if *size <= target_size {
                // Target level — apply the edit
                let new_sdf = edit_fn(sdf.clone());
                let new_node = OctreeNode::get_node(*center, *size, &new_sdf);

                invalidate_cache(node, child_cache);

                if let Some((parent, octant)) = parent_and_octant {
                    if let Some(children) = child_cache.get_mut(parent) {
                        children[octant] = new_node;
                    }
                }
                return;
            }

            // Get children, then collect the ones that overlap the edit region
            let children = child_cache
                .entry(node.clone())
                .or_insert_with(|| node.get_child_nodes())
                .clone();

            for (i, child) in children.iter().enumerate() {
                if let OctreeNode::Node { center: cc, size: cs, .. } = child {
                    if aabb_overlap(*cc, cs / 2.0, edit_center, edit_half) {
                        edit_node_recursive(
                            child, child_cache, edit_center, edit_half,
                            target_size, edit_fn, Some((node, i)),
                        );
                    }
                }
            }
        }
    }
}

/// Ray-AABB intersection. Returns (t_enter, t_exit) or None if miss.
fn ray_aabb(pos: Vec3, dir: Vec3, center: Vec3, half: f32) -> Option<(f32, f32)> {
    let bmin = center - Vec3::new(half, half, half);
    let bmax = center + Vec3::new(half, half, half);
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    for i in 0..3 {
        let (p, d, lo, hi) = match i {
            0 => (pos.x, dir.x, bmin.x, bmax.x),
            1 => (pos.y, dir.y, bmin.y, bmax.y),
            _ => (pos.z, dir.z, bmin.z, bmax.z),
        };
        if d.abs() < 1e-12 {
            if p < lo || p > hi {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut t1 = (lo - p) * inv;
            let mut t2 = (hi - p) * inv;
            if t1 > t2 { std::mem::swap(&mut t1, &mut t2); }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return None;
            }
        }
    }
    Some((t_min, t_max))
}

/// Cast a ray using the octree as an acceleration structure.
/// Traverses the octree hierarchically, skipping empty space, and
/// sphere-traces only within non-empty leaf nodes.
pub fn fast_cast_ray(
    root: &OctreeNode,
    child_cache: &mut std::collections::HashMap<OctreeNode, [OctreeNode; 8]>,
    pos: Vec3,
    dir: Vec3,
    max_dist: f32,
    min_node_size: f32,
) -> Option<(f32, Vec3)> {
    // Stack entries: (node, t_enter) — sorted so nearest is popped first
    let mut stack: Vec<(OctreeNode, f32)> = Vec::with_capacity(64);

    match root {
        OctreeNode::Empty => return None,
        OctreeNode::Node { center, size, .. } => {
            if let Some((t_enter, _)) = ray_aabb(pos, dir, *center, size / 2.0) {
                stack.push((root.clone(), t_enter.max(0.0)));
            } else {
                return None;
            }
        }
    }
    let mut hits: Vec<(OctreeNode, f32)> = Vec::with_capacity(8);
                    
    while let Some((node, t_enter)) = stack.pop() {
        if t_enter > max_dist {
            continue;
        }

        match &node {
            OctreeNode::Empty => {}
            OctreeNode::Node { center, size, sdf } => {
                if *size <= min_node_size {
                    // Leaf node: sphere-trace within this AABB
                    let half = size / 2.0;
                    let (_, t_exit) = ray_aabb(pos, dir, *center, half).unwrap_or((0.0, 0.0));
                    let t_exit = t_exit.min(max_dist);
                    let mut t = t_enter;
                    for _ in 0..128 {
                        if t > t_exit + 0.01 {
                            break;
                        }
                        let p = pos + dir * t;
                        let d = sdf.distance(p);
                        if d < 0.01 {
                            return Some((t, p));
                        }
                        t += d.max(0.01);
                    }
                } else {
                    // Internal node: subdivide and push children sorted back-to-front
                    // (so front children are popped first)
                    let children = child_cache
                        .entry(node.clone())
                        .or_insert_with(|| node.get_child_nodes());
                    hits.clear();
                    // Collect children that the ray hits
                    for child in children.iter() {
                        match child {
                            OctreeNode::Empty => {}
                            OctreeNode::Node { center: cc, size: cs, .. } => {
                                if let Some((t_en, _)) = ray_aabb(pos, dir, *cc, cs / 2.0) {
                                    let t_en = t_en.max(0.0);
                                    if t_en <= max_dist {
                                        hits.push((child.clone(), t_en));
                                    }
                                }
                            }
                        }
                    }
                    // Sort back-to-front so nearest is popped last (stack order)
                    hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    stack.extend(hits.drain(..));
                }
            }
        }
    }
    None
}

/// Build an octree from scratch for an SDF (no old tree to diff against).
pub fn build_octree(sdf: &Sdf, root_size: f32) -> OctreeNode {
    let _reused_count = 0u32;

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
