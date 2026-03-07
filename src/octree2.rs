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
    /// Fully inside the geometry — solid block, no visible surface.
    Solid {
        center: Vec3,
        size: f32,
    },
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
                let children : [OctreeNode;8] = std::array::from_fn(|i| {
                    let child_center = *center + octant_offset(i, child_size / 2.0);
                    OctreeNode::get_node(child_center, child_size, sdf)
                });
            
                children
            }
            OctreeNode::Empty => {
                std::array::from_fn(|_i| { OctreeNode::Empty})
            }
            OctreeNode::Solid { center, size } => {
                let child_size = size / 2.0;
                std::array::from_fn(|i| {
                    let child_center = *center + octant_offset(i, child_size / 2.0);
                    OctreeNode::Solid { center: child_center, size: child_size }
                })
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
        let center_distance = optimized.distance(center); 
        if center_distance > half_diag {
            return OctreeNode::Empty;
        }

        if center_distance < -half_diag {
            return OctreeNode::Solid { center, size };
        }
        
        // Corner check for Solid: sample all 8 corners.
        // Threshold: voxelizer needs d < -step*0.8, worst-case voxel offset
        // from corner is 0.5*step*sqrt(3), so corner needs d < -size*0.42.
        // We use -size*0.40 (slightly less conservative).
        let h = size * 0.5;
        let inside_threshold = -size * 0.40;
        let mut all_inside = true;
        for i in 0..8usize {
            let corner = center + octant_offset(i, h);
            if optimized.distance(corner) > inside_threshold {
                all_inside = false;
                break;
            }
        }
        if all_inside {
            return OctreeNode::Solid { center, size };
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
            OctreeNode::Solid { .. } => return current,
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
        OctreeNode::Empty | OctreeNode::Solid { .. } => return,
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
        OctreeNode::Empty | OctreeNode::Solid { .. } => return None,
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
            OctreeNode::Empty | OctreeNode::Solid { .. } => {}
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
                            OctreeNode::Empty | OctreeNode::Solid { .. } => {}
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
            OctreeNode::Solid { center, size } => {
                2u8.hash(state);
                center.x.to_bits().hash(state);
                center.y.to_bits().hash(state);
                center.z.to_bits().hash(state);
                size.to_bits().hash(state);
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
            (
                OctreeNode::Solid { center: c1, size: s1 },
                OctreeNode::Solid { center: c2, size: s2 },
            ) => c1.x.to_bits() == c2.x.to_bits()
                && c1.y.to_bits() == c2.y.to_bits()
                && c1.z.to_bits() == c2.z.to_bits()
                && s1.to_bits() == s2.to_bits(),
            _ => false,
        }
    }
}

impl Eq for OctreeNode {}

#[cfg(test)]
mod tests {
    use super::*;
    use supersdf::sdf::*;

    /// Place an AABB box on top of a small sphere and verify that octree nodes
    /// around but not touching the geometry are classified as Empty, not Node.
    #[test]
    fn test_no_spurious_nodes_around_box_on_sphere() {
        // Small sphere at origin + box sitting on top
        let sphere = Sdf::sphere(Vec3::new(0.0, 0.0, 0.0), 5.0);
        let boxx = Sdf::aabb(Vec3::new(0.0, 10.0, 0.0), Vec3::new(4.0, 4.0, 4.0));
        let sdf = Add::new(sphere, boxx).into();

        let root_size = 64.0;
        let root = build_octree(&sdf, root_size);

        // Walk the octree down to leaf-ish nodes and check for spurious Nodes
        let mut stack = vec![root];
        let mut spurious = Vec::new();
        while let Some(node) = stack.pop() {
            match &node {
                OctreeNode::Empty | OctreeNode::Solid { .. } => {}
                OctreeNode::Node { center, size, sdf: node_sdf } => {
                    if *size <= 4.0 {
                        // At leaf size, check if the SDF is actually anywhere
                        // near this node. If the closest surface is far away,
                        // this node should have been pruned to Empty.
                        let half = size / 2.0;
                        let d = node_sdf.distance(*center);
                        // If SDF distance at center is more than half-diagonal,
                        // no surface can intersect this cube — it's spurious.
                        let half_diag = size * 0.5 * 1.7320508;
                        if d > half_diag {
                            spurious.push((*center, *size, d));
                        }
                    } else {
                        let children = node.get_child_nodes();
                        for child in children {
                            stack.push(child);
                        }
                    }
                }
            }
        }
        if !spurious.is_empty() {
            for (center, size, d) in &spurious {
                println!(
                    "SPURIOUS Node at center={:?} size={} distance_at_center={}",
                    center, size, d
                );
            }
            panic!(
                "Found {} spurious Node(s) that should be Empty",
                spurious.len()
            );
        }
    }

    /// Check that nodes well inside the ground slab are Solid, not Node.
    #[test]
    fn test_ground_interior_is_solid() {
        let ground = Sdf::aabb(
            Vec3::new(0.0, -1010.0, 0.0),
            Vec3::new(10000.0, 1000.0, 10000.0),
        );
        let node = OctreeNode::get_node(
            Vec3::new(0.0, -1010.0, 0.0),
            64.0,
            &ground,
        );
        assert!(
            matches!(node, OctreeNode::Solid { .. }),
            "Node deep inside ground should be Solid, got {:?}",
            match &node {
                OctreeNode::Node { center, size, .. } => format!("Node(center={:?}, size={})", center, size),
                OctreeNode::Empty => "Empty".to_string(),
                OctreeNode::Solid { .. } => "Solid".to_string(),
            }
        );
    }

    /// Reproduce the real scene: ground slab + spheres on top.
    /// Walk the octree at super-chunk granularity (size 64 with min_size 4)
    /// and check which leaf nodes would produce empty voxelizations.
    #[test]
    fn test_scene_no_empty_superchunks() {
        let ground = Sdf::aabb(
            Vec3::new(0.0, -1010.0, 0.0),
            Vec3::new(10000.0, 1000.0, 10000.0),
        );
        let sphere = Sdf::sphere(Vec3::new(0.0, 0.0, 0.0), 20.0);
        let sdf: Sdf = Add::new(ground, sphere).into();
        let sdf = sdf.optimized_for_block(Vec3::ZERO, 2048.0);

        let root = build_octree(&sdf, 2048.0);
        let min_size = 4.0;

        let mut stack = vec![root];
        let mut total_leaves = 0u32;
        let mut no_surface = 0u32;
        let mut mostly_inside = 0u32;
        while let Some(node) = stack.pop() {
            match &node {
                OctreeNode::Empty | OctreeNode::Solid { .. } => {}
                OctreeNode::Node { center, size, sdf: node_sdf } => {
                    if *size <= min_size {
                        total_leaves += 1;
                        // Count voxels by type
                        let step = size / 4.0;
                        let half = size / 2.0;
                        let mut inside = 0u32;
                        let mut surface = 0u32;
                        let mut outside = 0u32;
                        for z in 0..4 {
                            for y in 0..4 {
                                for x in 0..4 {
                                    let pt = Vec3::new(
                                        center.x - half + (x as f32 + 0.5) * step,
                                        center.y - half + (y as f32 + 0.5) * step,
                                        center.z - half + (z as f32 + 0.5) * step,
                                    );
                                    let d = node_sdf.distance(pt);
                                    if d < -step * 0.8 {
                                        inside += 1;
                                    } else if d < step * 0.8 {
                                        surface += 1;
                                    } else {
                                        outside += 1;
                                    }
                                }
                            }
                        }
                        if surface == 0 {
                            no_surface += 1;
                            let d_center = node_sdf.distance(*center);
                            let h = size / 2.0;
                            print!(
                                "NO-SURFACE leaf: center=({:.1}, {:.1}, {:.1}) size={} inside={} outside={} d_center={:.3} corners=[",
                                center.x, center.y, center.z, size, inside, outside, d_center
                            );
                            for ci in 0..8usize {
                                let corner = *center + octant_offset(ci, h);
                                let d = node_sdf.distance(corner);
                                print!("{:.3}, ", d);
                            }
                            println!("]");
                        }
                        if surface > 0 && surface <= 4 && inside > 50 {
                            mostly_inside += 1;
                        }
                    } else {
                        let children = node.get_child_nodes();
                        for child in children {
                            stack.push(child);
                        }
                    }
                }
            }
        }
        println!("Total leaf nodes: {}", total_leaves);
        println!("Nodes with NO surface voxels (would be empty chunks): {}", no_surface);
        println!("Nodes mostly inside (>50 inside, <=4 surface): {}", mostly_inside);
        assert_eq!(
            no_surface, 0,
            "Found {} leaf nodes with no surface — should have been pruned to Empty or Solid",
            no_surface
        );
    }

    /// Test wall detection at various octree levels, simulating the
    /// real traversal from root down to leaf.
    #[test]
    fn test_wall_detection_ground() {
        let ground = Sdf::aabb(
            Vec3::new(0.0, -1010.0, 0.0),
            Vec3::new(10000.0, 1000.0, 10000.0),
        );
        // Ground surface at y = -10 (top) and y = -2010 (bottom).
        let test_cases: Vec<(Vec3, f32, &str, bool)> = vec![
            (Vec3::new(0.0, -64.0, 0.0), 128.0, "Y plane at y=-64 (in ground)", true),
            (Vec3::new(0.0, 0.0, 0.0), 128.0, "Y plane at y=0 (above surface)", false),
            (Vec3::new(0.0, -10.0, 0.0), 8.0, "Y plane at y=-10 (at surface)", false),
            (Vec3::new(0.0, -20.0, 0.0), 8.0, "Y plane at y=-20 (below surface)", true),
        ];

        for (center, size, desc, expect_solid) in &test_cases {
            let opt = ground.optimized_for_block(*center, *size);
            let half = size / 2.0;
            let grid_n = 3usize;
            let grid_spacing = size / grid_n as f32;
            let threshold = -grid_spacing * 0.7072;

            let mut solid = true;
            for iu in 0..grid_n {
                for iv in 0..grid_n {
                    let u = center.x - half + (iu as f32 + 0.5) * grid_spacing;
                    let v = center.z - half + (iv as f32 + 0.5) * grid_spacing;
                    let pt = Vec3::new(u, center.y, v);
                    let d = opt.distance(pt);
                    if d > threshold {
                        println!("  {} — sample ({:.1}, {:.1}, {:.1}) d={:.3} > threshold {:.3}",
                            desc, pt.x, pt.y, pt.z, d, threshold);
                        solid = false;
                    }
                }
            }
            println!("{}: solid={} (expected {})", desc, solid, expect_solid);
            assert_eq!(solid, *expect_solid, "Wall check failed for: {}", desc);
        }
    }

    /// Test wall detection using the actual optimized SDFs stored in octree nodes,
    /// as the traversal code would use them.
    #[test]
    fn test_wall_detection_with_octree_sdf() {
        let ground = Sdf::aabb(
            Vec3::new(0.0, -1010.0, 0.0),
            Vec3::new(10000.0, 1000.0, 10000.0),
        );
        let sphere = Sdf::sphere(Vec3::new(0.0, 0.0, 0.0), 20.0);
        let sdf: Sdf = Add::new(ground, sphere).into();
        let sdf = sdf.optimized_for_block(Vec3::ZERO, 2048.0);

        let root = build_octree(&sdf, 2048.0);

        // Walk the octree; at each Node check if the Y-wall detection using
        // the node's stored (optimized) SDF matches the real SDF.
        let mut stack = vec![root];
        let mut wall_found = 0u32;
        let mut wall_missed = 0u32;
        while let Some(node) = stack.pop() {
            match &node {
                OctreeNode::Empty | OctreeNode::Solid { .. } => {}
                OctreeNode::Node { center, size, sdf: node_sdf } => {
                    if *size <= 4.0 { continue; }

                    let half = size / 2.0;
                    let grid_n = 3usize;
                    let grid_spacing = size / grid_n as f32;
                    let threshold = -grid_spacing * 0.7072;

                    let mut y_solid = true;
                    for iu in 0..grid_n {
                        for iv in 0..grid_n {
                            let u = center.x - half + (iu as f32 + 0.5) * grid_spacing;
                            let v = center.z - half + (iv as f32 + 0.5) * grid_spacing;
                            let pt = Vec3::new(u, center.y, v);
                            let d = node_sdf.distance(pt);
                            if d > threshold {
                                y_solid = false;
                                let real_d = sdf.distance(pt);
                                if real_d < threshold {
                                    println!(
                                        "MISMATCH at ({:.0},{:.0},{:.0}) size={}: node_sdf={:.2} real={:.2} thresh={:.2}",
                                        center.x, center.y, center.z, size, d, real_d, threshold
                                    );
                                    wall_missed += 1;
                                }
                            }
                        }
                    }
                    if y_solid { wall_found += 1; }

                    let children = node.get_child_nodes();
                    for child in children {
                        stack.push(child);
                    }
                }
            }
        }
        println!("Y-walls found: {}, Mismatches (node_sdf wrong): {}", wall_found, wall_missed);
        assert_eq!(wall_missed, 0, "Node's optimized SDF disagrees with real SDF on {} planes", wall_missed);
    }
}
