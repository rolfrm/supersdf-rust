
use rand::{thread_rng};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fmt::Display;
use std::fmt;
use std::rc::Rc;

use crate::color::Color;
use crate::vec3::Vec3;

const SQRT3: f32 = 1.73205080757;

pub fn mix(a: i32, b: i32) -> i32{
        // murmurhash3-style finalizer mixing two inputs
        let mut h = 0x811c9dc5u32;
        h = h.wrapping_mul(0x01000193);
        h ^= a as u32;
        h = h.wrapping_mul(0x01000193);
        h ^= b as u32;
        // avalanche (murmurhash3 fmix32)
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        h as i32
    }

#[derive(Clone, Debug, PartialEq)]
pub enum Primitive{
    Sphere(Sphere),
    Aabb(Aabb)
}

impl Primitive{
    fn distance(&self, p : Vec3) -> f32 {
        
        match self{
            Primitive::Sphere(s) => s.distance(p),
            Primitive::Aabb(s) => s.distance(p),
        }
    }

}

#[derive(Clone, Debug, PartialEq)]
pub enum Coloring{
    SolidColor(Color),
    Gradient(Gradient),
    Noise(Noise),
    ColorScale(ColorScale)
}

impl Coloring {
    pub fn from_color(color : Color) -> Coloring {
        Coloring::SolidColor(color)
    }
    pub fn color(&self, p : Vec3) ->Color {
        match self {
            Coloring::SolidColor(c) => *c,
            Coloring::Gradient(g) => g.color(p),
            Coloring::Noise(n) => n.color(p),
            Coloring::ColorScale(s) => s.color(p),
        }
    }
}

#[derive(Clone, Debug)]
pub enum DistanceFieldEnum {
    Primitive(Primitive),
    Coloring(Coloring, Rc<DistanceFieldEnum>),
    Add(Add),
    Subtract(Subtract),
    Empty
}

pub trait DistanceField {
    fn distance(&self, _pos: Vec3) -> f32 {
        return f32::INFINITY;
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Sphere {
    pub(crate) center: Vec3,
    pub(crate) radius: f32
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32) -> Sphere {
        Sphere {
            center: center,
            radius: radius
        }
    }

    pub fn color(self, color : Color) -> DistanceFieldEnum {
        let out: DistanceFieldEnum = self.into();
        return out.colored(color)
    }

    pub fn two_sphere_bounds(a: &Sphere, b: &Sphere) -> Sphere {
        // If either has non-finite radius, return the other (or keep infinite)
        if !a.radius.is_finite() { return b.clone(); }
        if !b.radius.is_finite() { return a.clone(); }

        let d = (a.center - b.center).length();

        // One sphere contains the other
        if d + b.radius <= a.radius { return a.clone(); }
        if d + a.radius <= b.radius { return b.clone(); }

        if d <= 0.000001 {
            return match a.radius < b.radius {
                true => b,
                false => a,
            }.clone();
        }

        let dir = (b.center - a.center) / d;
        // Extreme points of the union along the connecting axis
        let leftext = a.center - dir * a.radius;
        let rightext = b.center + dir * b.radius;

        let center = (leftext + rightext) * 0.5;
        let radius = (leftext - rightext).length() * 0.5;

        Sphere::new(center, radius)
    }

    pub fn overlaps(&self, other: &Sphere) -> bool {
        let combined_radius = self.radius + other.radius;
        (self.center - other.center).length_squared() < combined_radius * combined_radius
    }

    /// Test if this sphere fits entirely inside an axis-aligned box.
    pub fn contained_in_aabb(&self, box_center: Vec3, box_half: f32) -> bool {
        (self.center.x - self.radius >= box_center.x - box_half)
            && (self.center.x + self.radius <= box_center.x + box_half)
            && (self.center.y - self.radius >= box_center.y - box_half)
            && (self.center.y + self.radius <= box_center.y + box_half)
            && (self.center.z - self.radius >= box_center.z - box_half)
            && (self.center.z + self.radius <= box_center.z + box_half)
    }

    /// Test if this sphere overlaps an axis-aligned box defined by center and half-size.
    pub fn overlaps_aabb(&self, box_center: Vec3, box_half: f32) -> bool {
        // Closest point on AABB to sphere center
        let dx = (self.center.x - box_center.x).clamp(-box_half, box_half);
        let dy = (self.center.y - box_center.y).clamp(-box_half, box_half);
        let dz = (self.center.z - box_center.z).clamp(-box_half, box_half);
        let closest = Vec3::new(box_center.x + dx, box_center.y + dy, box_center.z + dz);
        (self.center - closest).length_squared() <= self.radius * self.radius
    }
}

impl Into<DistanceFieldEnum> for Sphere {
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Primitive(Primitive::Sphere(self))
    }
}

fn vec3_max(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(f32::max(a.x, b.x), f32::max(a.y, b.y), f32::max(a.z, b.z))
}

fn vec3_min(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(f32::min(a.x, b.x), f32::min(a.y, b.y), f32::min(a.z, b.z))
}

#[derive(Clone, Debug)]
pub struct AabbBounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl AabbBounds {
    pub fn new(min: Vec3, max: Vec3) -> AabbBounds {
        AabbBounds { min, max }
    }

    pub fn infinite() -> AabbBounds {
        AabbBounds {
            min: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
            max: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
        }
    }

    pub fn is_finite(&self) -> bool {
        self.min.x.is_finite() && self.min.y.is_finite() && self.min.z.is_finite()
            && self.max.x.is_finite() && self.max.y.is_finite() && self.max.z.is_finite()
    }

    pub fn union(&self, other: &AabbBounds) -> AabbBounds {
        AabbBounds {
            min: vec3_min(self.min, other.min),
            max: vec3_max(self.max, other.max),
        }
    }

    /// Test if this AABB overlaps a cube defined by center and half-size.
    pub fn overlaps_aabb(&self, box_center: Vec3, box_half: f32) -> bool {
        let bmin = box_center - Vec3::new(box_half, box_half, box_half);
        let bmax = box_center + Vec3::new(box_half, box_half, box_half);
        self.min.x <= bmax.x && self.max.x >= bmin.x
            && self.min.y <= bmax.y && self.max.y >= bmin.y
            && self.min.z <= bmax.z && self.max.z >= bmin.z
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Aabb {
    pub(crate) radius: Vec3,
    pub(crate) center: Vec3
}

impl Aabb {
    pub fn new(center: Vec3, radius: Vec3) -> Aabb {
        Aabb {
            radius: radius,
            center: center
        }
    }

    pub fn distance(&self, p: Vec3) -> f32 {
        let p2 = p - self.center;
        let q = p2.abs() - self.radius;
        return vec3_max(q, Vec3::zeros()).length()
            + f32::min(f32::max(q.x, f32::max(q.y, q.z)), 0.0);
    }
}

impl Into<DistanceFieldEnum> for Aabb {
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Primitive(Primitive::Aabb(self))
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Gradient {
    pub(crate) p1: Vec3,
    pub(crate) p2: Vec3,
    pub(crate) c1: Color,
    pub(crate) c2: Color,
}

impl Gradient {
    pub fn new(
        p1: Vec3,
        p2: Vec3,
        c1: Color,
        c2: Color,
        inner: Rc<DistanceFieldEnum>,
    ) -> DistanceFieldEnum {
        DistanceFieldEnum::Coloring(Coloring::Gradient(Gradient {
            p1: p1,
            p2: p2,
            c1: c1,
            c2: c2,
        }),inner)
    }

    pub fn color(&self, p: Vec3) -> Color {
        let pt2 = p - self.p1;
        let l2 = (self.p1 - self.p2).length_squared();
        let f = (self.p2 - self.p1).dot(pt2) / l2;
        let color = rgba_interp(self.c1, self.c2, f);
        return color;
    }
}


#[derive(Clone, Debug)]
pub struct Noise {
    //noise: Rc<Perlin>,
    pub(crate) seed: u32,
    pub(crate) c1: Color,
    pub(crate) c2: Color
}


impl fmt::Display for Noise{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Perlin {}", self.c1)?;
        write!(f, " {}", self.c2)
    }
}

impl PartialEq for Noise {
    fn eq(&self, other: &Self) -> bool {
        return self.seed == other.seed
            && self.c1 == other.c1
            && self.c2 == other.c2
    }
}

impl Noise {
    pub fn new(seed: u32, c1: Color, c2: Color, inner: DistanceFieldEnum) -> DistanceFieldEnum {
        DistanceFieldEnum::Coloring(Coloring::Noise(Noise {
            seed: seed,
            //noise: Rc::new(Perlin::new(seed)),
            c1: c1,
            c2: c2
        }), Rc::new(inner))
    }
    pub fn new2(seed: u32, c1: Color, c2: Color) -> Coloring {
        Coloring::Noise(Noise {
            seed: seed,
            //noise: Rc::new(Perlin::new(seed)),
            c1: c1,
            c2: c2
        })
    }
    fn color(&self, pos: Vec3) -> Color {
        let ix = (pos.x * 1.0) as i32;
        let iy = (pos.y * 1.0) as i32;
        let iz = (pos.z * 1.0) as i32;
        let h = mix(mix(ix, iy), mix(iz, self.seed as i32));
        let t = ((h & 0xFFFF) as f32) / 65535.0;
        rgba_interp(self.c1, self.c2, t)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct ColorScale {
    pub(crate) scale: Vec3,
    pub(crate) inner: Rc<Coloring>,
}

impl ColorScale {
    pub fn new(scale: Vec3, inner_coloring: Coloring, inner_df: Rc<DistanceFieldEnum>) -> DistanceFieldEnum {
        DistanceFieldEnum::Coloring(Coloring::ColorScale(ColorScale {
            scale,
            inner: Rc::new(inner_coloring),
        }), inner_df)
    }

    fn color(&self, p: Vec3) -> Color {
        let scaled = Vec3::new(p.x * self.scale.x, p.y * self.scale.y, p.z * self.scale.z);
        self.inner.color(scaled)
    }
}


#[derive(Clone, PartialEq, Debug)]
pub struct Add {
    pub(crate) items: Vec<Rc<DistanceFieldEnum>>,
    pub(crate) bounds: Sphere,

}
fn rgba_interp(a: Color, b: Color, v: f32) -> Color {

    fn interp(av : f32, bv : f32, amount : f32) -> f32 {
        av * (1.0 - amount) + bv * amount
    }
    Color::rgba(
        interp(a.r, b.r, v),
        interp(a.g, b.g, v),
        interp(a.b, b.b, v),
        interp(a.a, b.a, v))
    
}
impl Add {
    pub fn new(left: DistanceFieldEnum, right: DistanceFieldEnum) -> Add {
        Add::from_items(vec![Rc::new(left), Rc::new(right)])
    }

    pub fn new2(left: DistanceFieldEnum, right: DistanceFieldEnum) -> DistanceFieldEnum {
        DistanceFieldEnum::Add(Add::from_items(vec![Rc::new(left), Rc::new(right)]))
    }

    pub fn from_rc(left: Rc<DistanceFieldEnum>, right: Rc<DistanceFieldEnum>) -> Add {
        Add::from_items(vec![left, right])
    }

    pub fn from_items_subdivide(items: Vec<Rc<DistanceFieldEnum>>, n: usize) -> Add {
        if items.len() <= n * n * n * 2 || n < 2 {
            return Add::from_items(items);
        }

        // Compute centers once
        let centers: Vec<Vec3> = items.iter()
            .map(|i| i.calculate_sphere_bounds().center)
            .collect();

        // Compute AABB of all centers
        let mut aabb_min = centers[0];
        let mut aabb_max = centers[0];
        for c in &centers[1..] {
            aabb_min = vec3_min(aabb_min, *c);
            aabb_max = vec3_max(aabb_max, *c);
        }

        let extent = aabb_max - aabb_min;
        // Add small epsilon to avoid items on the max edge falling out of bounds
        let cell_size = Vec3::new(
            if extent.x > 0.0 { extent.x / n as f32 } else { 1.0 },
            if extent.y > 0.0 { extent.y / n as f32 } else { 1.0 },
            if extent.z > 0.0 { extent.z / n as f32 } else { 1.0 },
        );

        // Bin items into grid cells
        let mut bins: Vec<Vec<Rc<DistanceFieldEnum>>> = vec![Vec::new(); n * n * n];
        for (idx, item) in items.into_iter().enumerate() {
            let rel = centers[idx] - aabb_min;
            let ix = ((rel.x / cell_size.x) as usize).min(n - 1);
            let iy = ((rel.y / cell_size.y) as usize).min(n - 1);
            let iz = ((rel.z / cell_size.z) as usize).min(n - 1);
            bins[ix * n * n + iy * n + iz].push(item);
        }

        // Build top-level items from non-empty bins
        let mut top_items: Vec<Rc<DistanceFieldEnum>> = Vec::new();
        for bin in bins {
            match bin.len() {
                0 => {},
                1 => top_items.push(bin.into_iter().next().unwrap()),
                _ => top_items.push(Rc::new(DistanceFieldEnum::Add(Add::from_items_subdivide(bin, n)))),
            }
        }

        Add::from_items(top_items)
    }

    pub fn from_items(mut items: Vec<Rc<DistanceFieldEnum>>) -> Add {
        // Sort items by bounds center (x, then y, then z) for deterministic ordering
        items.sort_by(|a, b| {
            let ab = a.calculate_sphere_bounds();
            let bb = b.calculate_sphere_bounds();
            ab.center.x.partial_cmp(&bb.center.x).unwrap()
                .then(ab.center.y.partial_cmp(&bb.center.y).unwrap())
                .then(ab.center.z.partial_cmp(&bb.center.z).unwrap())
        });

        let bounds = items.iter()
            .map(|i| i.calculate_sphere_bounds())
            .reduce(|a, b| Sphere::two_sphere_bounds(&a, &b))
            .unwrap_or_else(|| Sphere::new(Vec3::zeros(), f32::INFINITY));
        Add {
            items,
            bounds,
        }
    }

    fn distance(&self, pos: Vec3) -> f32 {
        let d1 = self.bounds.distance(pos);
        if d1 > self.bounds.radius {
            return d1;
        }

        self.items.iter()
            .map(|item| item.distance(pos))
            .fold(f32::INFINITY, f32::min)
    }

    fn color(&self, p: Vec3) -> Color {
        let mut best_d = f32::INFINITY;
        let mut best_color = Color::TRANSPARENT;
        for item in &self.items {
            let d = item.distance(p);
            if d < best_d {
                best_d = d;
                best_color = item.color(p);
            }
        }
        best_color
    }
}

impl Into<DistanceFieldEnum> for Add {
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Add(self)
    }
}

impl DistanceField for Sphere {
    fn distance(&self, pos: Vec3) -> f32 {
        (pos - self.center).length() - self.radius
    }
}
impl DistanceField for DistanceFieldEnum {
    fn distance(&self, pos: Vec3) -> f32 {
        match self {
            DistanceFieldEnum::Add(add) => add.distance(pos),
            DistanceFieldEnum::Subtract(sub) => sub.distance(pos),
            DistanceFieldEnum::Primitive(primitive) => primitive.distance(pos),
            DistanceFieldEnum::Empty => f32::INFINITY,
            DistanceFieldEnum::Coloring(_, inner) => inner.distance(pos)
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Subtract {
     pub(crate) left : Rc<DistanceFieldEnum>,
     pub(crate) subtract: Rc<DistanceFieldEnum>,
     pub(crate) k : f32
}

fn f32mixf( x : f32,  y : f32,  a : f32) -> f32 { x * (1.0 - a) + y * a }

impl DistanceField for Subtract {
    fn distance(&self, pos : Vec3) -> f32 {
    let k = self.k;
    let d1 = self.left.distance(pos);
    let d2 = self.subtract.distance(pos);
    let h = f32::clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    
    let r = f32mixf(d1, -d2, h) + k * h * (1.0 - h);
    return r;
    }
}

impl Subtract {
    pub fn new<T : Into<DistanceFieldEnum>, T2 : Into<DistanceFieldEnum>>(left : T, subtract : T2, k : f32) -> Subtract {
        Subtract { left: Rc::new(left.into()), subtract: Rc::new(subtract.into()), k: k }
    }
}
impl Into<DistanceFieldEnum> for Subtract {
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Subtract(self)
    }
}

impl DistanceFieldEnum {

    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
    
    
    pub fn colored(&self, color: Color) -> DistanceFieldEnum {
        DistanceFieldEnum::Coloring(Coloring::SolidColor(color), Rc::new(self.clone()))
    }

    pub fn sphere(pos : Vec3, radius : f32) -> DistanceFieldEnum {
        Sphere::new(pos, radius).into()
    }

    pub fn aabb(pos: Vec3, size: Vec3) -> DistanceFieldEnum {
        Aabb::new(pos, size).into()
    }

    pub fn add(&self, sdf : DistanceFieldEnum) -> DistanceFieldEnum {
        DistanceFieldEnum::Add(Add::new(self.clone(), sdf.clone()))
    }

    pub fn subtract(&self, sdf : DistanceFieldEnum) -> DistanceFieldEnum {
        DistanceFieldEnum::Subtract(Subtract::new(self.clone(), sdf, 0.1))
    }

    pub fn cast_ray(&self, pos: Vec3, dir: Vec3, max_dist: f32) -> Option<(f32, Vec3)> {
        let mut total_distance = 0.0;
        let mut mpos = pos;
        for _ in 0..512 {
            let d = self.distance(mpos);

            total_distance += d;
            mpos = mpos + dir * d;
            if total_distance > max_dist {
                return None;
            }
            if d < 0.01 {
                return Some((total_distance, mpos));
            }
        }
        None
    }

    pub fn optimize_add(add: &Add, block_center: Vec3, size: f32) -> Option<DistanceFieldEnum> {
        let mut optimized: Vec<(Rc<DistanceFieldEnum>, f32)> = Vec::new();
        let mut any_changed = false;
        for item in add.items.iter() {
            match item.optimized_for_block2(block_center, size) {
                Some(DistanceFieldEnum::Empty) => { any_changed = true; },
                Some(opt) => {
                    any_changed = true;
                    let d = opt.distance(block_center);
                    optimized.push((Rc::new(opt), d));
                },
                None => {
                    let d = item.distance(block_center);
                    optimized.push((item.clone(), d));
                },
            }
        }

        if optimized.is_empty() {
            return Some(DistanceFieldEnum::Empty);
        }
        if optimized.len() == 1 {
            let (rc, _) = optimized.into_iter().next().unwrap();
            return Some(Rc::try_unwrap(rc).unwrap_or_else(|rc| rc.as_ref().clone()));
        }

        // Filter items that can't contribute to the union surface in this block.
        // For Lipschitz-1 SDFs: item B can't beat item A anywhere in the block if
        // B's distance at center > A's distance at center + block diagonal.
        let block_diag = size * SQRT3;
        let closest_d = optimized.iter().map(|(_, d)| *d).fold(f32::INFINITY, f32::min);
        let prev_len = optimized.len();
        optimized.retain(|(_, d)| *d <= closest_d + block_diag);

        if optimized.len() != prev_len {
            any_changed = true;
        }
        if optimized.len() == 1 {
            let (rc, _) = optimized.into_iter().next().unwrap();
            return Some(Rc::try_unwrap(rc).unwrap_or_else(|rc| rc.as_ref().clone()));
        }

        if !any_changed {
            return None;
        }

        let items: Vec<Rc<DistanceFieldEnum>> = optimized.into_iter().map(|(rc, _)| rc).collect();
        Some(DistanceFieldEnum::Add(Add::from_items(items)))
    }
    pub fn optimized_for_block(&self, block_center: Vec3, size: f32) -> Rc<DistanceFieldEnum> {
        match self.optimized_for_block2(block_center, size) {
            Some(opt) => Rc::new(opt),
            None => Rc::new(self.clone()),
        }
    }

    pub fn optimized_for_block2(&self, block_center: Vec3, size: f32) -> Option<DistanceFieldEnum> {
        let half = size / 2.0;
        match self {
            DistanceFieldEnum::Add(add) => {
                // If the Add's bounding sphere doesn't overlap the block at all, eliminate it
                let b = &add.bounds;
                if !b.overlaps_aabb(block_center, half) {
                    return Some(DistanceFieldEnum::Empty);
                }
                if b.contained_in_aabb(block_center, half) {
                    return None;
                }
                
                DistanceFieldEnum::optimize_add(add, block_center, size)
            },
            DistanceFieldEnum::Subtract(sub) => {
                let optsub = sub.subtract.optimized_for_block2(block_center, size);
                let optsub_ref = optsub.as_ref().map_or(sub.subtract.as_ref(), |v| v);

                let subtract_d = optsub_ref.distance(block_center);
                let optleft = sub.left.optimized_for_block2(block_center, size);
                if subtract_d > half * SQRT3 * 2.0 {
                    // Subtractor is too far away, strip the Subtract wrapper
                    return Some(match optleft {
                        Some(left) => left,
                        None => sub.left.as_ref().clone(),
                    });
                }

                match (&optleft, &optsub) {
                    (None, None) => None,
                    (None, Some(s)) => Some(DistanceFieldEnum::Subtract(Subtract {
                        left: sub.left.clone(), subtract: Rc::new(s.clone()), k: sub.k
                    })),
                    (Some(l), None) => Some(DistanceFieldEnum::Subtract(Subtract {
                        left: Rc::new(l.clone()), subtract: sub.subtract.clone(), k: sub.k
                    })),
                    (Some(l), Some(s)) => Some(DistanceFieldEnum::Subtract(Subtract {
                        left: Rc::new(l.clone()), subtract: Rc::new(s.clone()), k: sub.k
                    })),
                }
            },
            DistanceFieldEnum::Coloring(c, i) => {
                match i.optimized_for_block2(block_center, size) {
                    Some(DistanceFieldEnum::Empty) => Some(DistanceFieldEnum::Empty),
                    Some(opt) => Some(DistanceFieldEnum::Coloring(c.clone(), Rc::new(opt))),
                    None => None,
                }
            }

            _ =>{
                let sb = self.calculate_sphere_bounds();
                if !sb.overlaps_aabb(block_center, half) {
                    return Some(DistanceFieldEnum::Empty);
                }
                None
            }
        }
    }

    pub fn distance_and_optimize(&self, pos: Vec3, size: f32) -> (f32, DistanceFieldEnum) {
        let pos2 : Vec3 = pos.map(|x| f32::floor(x / size) * size);
        let sdf2 =
            self.optimized_for_block(pos2 + Vec3::new(size * 0.5, size * 0.5, size * 0.5), size);
        
        return (sdf2.distance(pos), sdf2.as_ref().clone());
    }

    pub fn insert(&self, sdf: DistanceFieldEnum) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Empty => sdf,
            DistanceFieldEnum::Add (add) =>{
                let mut items2 : Vec<Rc<DistanceFieldEnum>> = (&add.items).into_iter().map(|i| i.clone()).collect();
                items2.push(Rc::new(sdf));
                Add::from_items(items2).into()
            },
            _ => Add::new(self.clone(), sdf).into(),
        }
    }
    
    pub fn fast_insert(&self, sdf: DistanceFieldEnum) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Empty => sdf,
            DistanceFieldEnum::Add (add) =>{
                let sdf_bounds = sdf.calculate_sphere_bounds();
                let center = sdf_bounds.center;

                let (index, item) = add.items.iter().enumerate()
                    .min_by_key(|(_, item)| {
                        let sb = item.calculate_sphere_bounds();
                        (((sb.center - center).length() - sb.radius) * 100.0).floor() as i32
                    })
                    .unwrap();

                let new_bounds = if sdf_bounds.radius.is_finite() {
                    Sphere::two_sphere_bounds(&add.bounds, &sdf_bounds)
                } else {
                    add.bounds.clone()
                };

                match item.as_ref() {
                    DistanceFieldEnum::Add(_child_add) => {
                        let new_child = item.fast_insert(sdf);
                        println!("sub");
                        let mut new_items = add.items.clone();
                        new_items[index] = new_child.into();
                        DistanceFieldEnum::Add(Add { items: new_items, bounds: new_bounds })
                    },
                    _ => {
                        let mut new_items = add.items.clone();
                        new_items.push(sdf.into());
                        println!(" found item: {}", item);
                        DistanceFieldEnum::Add(Add { items: new_items, bounds: new_bounds })
                    }
                }
            },
            _ => Add::new(self.clone(), sdf).into(),
        }
    }

    pub fn print_layout(&self, idx: i32){
        match self {
            DistanceFieldEnum::Add (add) => {
                println!("{} add {}", idx, add.items.len());
                for item in &(add.items) {
                    item.print_layout(idx + 1);
                }
            },
            _ => {/*println!(" {} - elem", idx);*/ }
        }
    }

    pub fn insert_3<T: Into<DistanceFieldEnum>>(&self, sdf: T, _bounds: &Sphere) -> DistanceFieldEnum {
        
        match self {
            DistanceFieldEnum::Add(_add) => {
                return self.insert(sdf.into());
            },
            _ => self.insert(sdf.into())
        }
    }

    pub fn insert_2<T: Into<DistanceFieldEnum>>(&self, sdf: T) -> DistanceFieldEnum {
        let sdf2 : DistanceFieldEnum = sdf.into().clone();
        let sphere = sdf2.calculate_sphere_bounds();
        return self.insert_3(sdf2, &sphere);
    }

    pub fn calculate_sphere_bounds(&self) -> Sphere {
        match self {
            DistanceFieldEnum::Primitive(p) =>{
                match p {
                    Primitive::Sphere(s) => s.clone(),
                    Primitive::Aabb(aabb) => Sphere::new(aabb.center, aabb.radius.length()),
                }
            },
            DistanceFieldEnum::Empty => Sphere::new(Vec3::zeros(), f32::INFINITY),
            DistanceFieldEnum::Coloring(_,inner) => inner.calculate_sphere_bounds(),
            DistanceFieldEnum::Add(add) => add.bounds.clone(),
            DistanceFieldEnum::Subtract(sub) => sub.left.calculate_sphere_bounds()
        }
    }

    pub fn calculate_aabb_bounds(&self) -> AabbBounds {
        match self {
            DistanceFieldEnum::Primitive(p) => match p {
                Primitive::Sphere(s) => {
                    let r = Vec3::new(s.radius, s.radius, s.radius);
                    AabbBounds::new(s.center - r, s.center + r)
                }
                Primitive::Aabb(aabb) => {
                    AabbBounds::new(aabb.center - aabb.radius, aabb.center + aabb.radius)
                }
            },
            DistanceFieldEnum::Empty => AabbBounds::infinite(),
            DistanceFieldEnum::Coloring(_, inner) => inner.calculate_aabb_bounds(),
            DistanceFieldEnum::Add(add) => {
                add.items.iter()
                    .map(|i| i.calculate_aabb_bounds())
                    .reduce(|a, b| a.union(&b))
                    .unwrap_or_else(AabbBounds::infinite)
            }
            DistanceFieldEnum::Subtract(sub) => sub.left.calculate_aabb_bounds(),
        }
    }

    pub fn optimize_bounds(&self) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Add(_) => {
                // 1. Flatten nested Adds recursively into a flat list
                fn flatten_add(sdf: &DistanceFieldEnum, out: &mut Vec<Rc<DistanceFieldEnum>>) {
                    match sdf {
                        DistanceFieldEnum::Add(add) => {
                            for item in &add.items {
                                flatten_add(item, out);
                            }
                        }
                        _ => out.push(Rc::new(sdf.clone())),
                    }
                }
                let mut flat_items: Vec<Rc<DistanceFieldEnum>> = Vec::new();
                flatten_add(self, &mut flat_items);

                // 2. Optimize each item recursively
                let flat_items: Vec<Rc<DistanceFieldEnum>> = flat_items.into_iter()
                    .map(|item| Rc::new(item.optimize_bounds()))
                    .collect();

                // 3. Compute sphere bounds for each
                let item_bounds: Vec<Sphere> = flat_items.iter()
                    .map(|i| i.calculate_sphere_bounds())
                    .collect();

                // 4. Group using union-find: merge groups where 50% bounds overlap
                let n = flat_items.len();
                let mut parent: Vec<usize> = (0..n).collect();

                fn uf_find(parent: &mut Vec<usize>, i: usize) -> usize {
                    if parent[i] != i {
                        parent[i] = uf_find(parent, parent[i]);
                    }
                    parent[i]
                }
                fn uf_union(parent: &mut Vec<usize>, i: usize, j: usize) {
                    let ri = uf_find(parent, i);
                    let rj = uf_find(parent, j);
                    if ri != rj { parent[ri] = rj; }
                }

                for i in 0..n {
                    for j in (i+1)..n {
                        let center_distance = (item_bounds[i].center - item_bounds[j].center).length();
                        if center_distance < (item_bounds[i].radius + item_bounds[j].radius) * 0.5 {
                            uf_union(&mut parent, i, j);
                        }
                    }
                }

                // Collect groups
                let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
                for i in 0..n {
                    let root = uf_find(&mut parent, i);
                    groups.entry(root).or_default().push(i);
                }

                // 5. Split groups > 10 items by proximity
                fn split_large_group(indices: Vec<usize>, bounds: &[Sphere]) -> Vec<Vec<usize>> {
                    if indices.len() <= 10 {
                        return vec![indices];
                    }
                    // Find longest axis
                    let centers: Vec<Vec3> = indices.iter().map(|&i| bounds[i].center).collect();
                    let min_c = centers.iter().fold(
                        Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
                        |a, b| vec3_min(a, *b));
                    let max_c = centers.iter().fold(
                        Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
                        |a, b| vec3_max(a, *b));
                    let extent = max_c - min_c;

                    let axis = if extent.x >= extent.y && extent.x >= extent.z { 0 }
                        else if extent.y >= extent.z { 1 }
                        else { 2 };

                    let mut sorted = indices;
                    sorted.sort_by(|&a, &b| {
                        let va = match axis { 0 => bounds[a].center.x, 1 => bounds[a].center.y, _ => bounds[a].center.z };
                        let vb = match axis { 0 => bounds[b].center.x, 1 => bounds[b].center.y, _ => bounds[b].center.z };
                        va.partial_cmp(&vb).unwrap()
                    });

                    let mid = sorted.len() / 2;
                    let left = sorted[..mid].to_vec();
                    let right = sorted[mid..].to_vec();

                    let mut result = split_large_group(left, bounds);
                    result.extend(split_large_group(right, bounds));
                    result
                }

                let mut final_groups: Vec<Vec<usize>> = Vec::new();
                for (_, group) in groups {
                    final_groups.extend(split_large_group(group, &item_bounds));
                }

                // 6. Build Add nodes from groups
                let mut group_nodes: Vec<Rc<DistanceFieldEnum>> = Vec::new();
                for group in final_groups {
                    if group.len() == 1 {
                        group_nodes.push(flat_items[group[0]].clone());
                    } else {
                        let items: Vec<Rc<DistanceFieldEnum>> = group.into_iter()
                            .map(|i| flat_items[i].clone())
                            .collect();
                        group_nodes.push(Rc::new(DistanceFieldEnum::Add(Add::from_items(items))));
                    }
                }

                if group_nodes.len() == 1 {
                    group_nodes.into_iter().next().unwrap().as_ref().clone()
                } else {
                    DistanceFieldEnum::Add(Add::from_items(group_nodes))
                }
            },
            DistanceFieldEnum::Subtract(sub) => {
                let left2 = sub.left.optimize_bounds();
                let subbounds = sub.subtract.calculate_sphere_bounds();

                // If the left side is an Add, try to push the subtraction into
                // only the item(s) whose bounds overlap the subtracted volume.
                if let DistanceFieldEnum::Add(inner_add) = &left2 {
                    let mut overlapping: Vec<Rc<DistanceFieldEnum>> = Vec::new();
                    let mut non_overlapping: Vec<Rc<DistanceFieldEnum>> = Vec::new();

                    for item in &inner_add.items {
                        let ib = item.calculate_sphere_bounds();
                        if ib.overlaps(&subbounds) {
                            overlapping.push(item.clone());
                        } else {
                            non_overlapping.push(item.clone());
                        }
                    }

                    if overlapping.is_empty() {
                        // No overlap -> subtraction has no effect
                        return DistanceFieldEnum::Add(inner_add.clone());
                    }

                    if !non_overlapping.is_empty() {
                        // Some items don't overlap -> push subtraction only into overlapping items
                        let overlap_sdf: DistanceFieldEnum = if overlapping.len() == 1 {
                            overlapping[0].as_ref().clone()
                        } else {
                            DistanceFieldEnum::Add(Add::from_items(overlapping))
                        };
                        let subtracted = Rc::new(
                            DistanceFieldEnum::Subtract(Subtract::new(
                                overlap_sdf,
                                sub.subtract.as_ref().clone(),
                                sub.k,
                            ))
                        );
                        let mut all_items = non_overlapping;
                        all_items.push(subtracted);
                        return DistanceFieldEnum::Add(Add::from_items(all_items));
                    }
                    // All overlap -> fall through to keep the subtraction on the whole Add
                }

                // Try to combine nested subtracts: Subtract(Subtract(A, B), C) → Subtract(A, Add(B, C))
                let mut new_left = Subtract::new(left2, sub.subtract.as_ref().clone(), sub.k).into();
                if let DistanceFieldEnum::Subtract(sub2) = &new_left {
                    if let DistanceFieldEnum::Subtract(sub3) = sub2.left.as_ref() {
                        let comb = sub3.subtract.insert_2(sub2.subtract.as_ref().clone())
                            .optimize_bounds();
                        new_left = Subtract::new(sub3.left.as_ref().clone(), comb, sub3.k).into();
                    }
                }

                new_left
            }
            _ => self.clone(),
        }
    }

    pub fn color(&self, pos: Vec3) -> Color {
        match self {
            DistanceFieldEnum::Add(add) => add.color(pos),
            DistanceFieldEnum::Primitive(_) => Color::RED,
            DistanceFieldEnum::Empty => Color::TRANSPARENT,
            DistanceFieldEnum::Coloring(c, _inner) => c.color(pos),
            DistanceFieldEnum::Subtract(sub) => sub.left.color(pos)
        }
    }

    pub fn gradient(&self, pos: Vec3, size: f32) -> Vec3 {
        let focus = self.optimized_for_block(pos, size);
        let mut ptx = pos;
        let mut pty = pos;
        let mut ptz = pos;
        ptx.x += size * 0.2;
        pty.y += size * 0.2;
        ptz.z += size * 0.2;
        let dx1 = focus.distance(ptx);
        let dy1 = focus.distance(pty);
        let dz1 = focus.distance(ptz);

        ptx.x -= size * 0.2 * 2.0;
        pty.y -= size * 0.2 * 2.0;
        ptz.z -= size * 0.2 * 2.0;
        let dx2 = focus.distance(ptx);
        let dy2 = focus.distance(pty);
        let dz2 = focus.distance(ptz);
        let dv = Vec3::new(dx1 - dx2, dy1 - dy2, dz1 - dz2);
        let l = dv.length();
        if l < 0.00001 {
            return Vec3::zeros();
        }
        let x = dv / l;
        return x;
    }

    pub fn equals(&self, other: &Self) -> bool {
        return match self {
            DistanceFieldEnum::Primitive(s) => {
                if let DistanceFieldEnum::Primitive(s2) = other {
                    return s.eq(s2);
                }
                false
            },
            DistanceFieldEnum::Add(a) => {
                if let DistanceFieldEnum::Add(b) = other {
                    if a.items.len() != b.items.len() { return false; }
                    return a.items.iter().zip(b.items.iter()).all(|(ai, bi)| {
                        Rc::ptr_eq(ai, bi) || ai.equals(bi)
                    });
                }
                false
            },
            DistanceFieldEnum::Coloring(a, i) => {
                if let DistanceFieldEnum::Coloring(b, i2) = other {
                    return  a.eq(b) && i.equals(i2);
                }
                false
            },
            DistanceFieldEnum::Subtract(a) => {
                if let DistanceFieldEnum::Subtract(b) = other {
                    return Rc::<DistanceFieldEnum>::ptr_eq(&a.left,&b.left) &&
                        Rc::<DistanceFieldEnum>::ptr_eq(&a.subtract,&b.subtract);
                }
                false

            },

            DistanceFieldEnum::Empty => {
                if let DistanceFieldEnum::Empty = other {
                    return true;
                }
                return false;

            },
        }
    }

    pub fn ref_eq(&self, other: &Self) -> bool {
        return match self {
            DistanceFieldEnum::Primitive(s) => {
                if let DistanceFieldEnum::Primitive(s2) = other {
                    return s.eq(s2);
                }
                false
            },
            DistanceFieldEnum::Add(a) => {
                if let DistanceFieldEnum::Add(b) = other {
                    if a.items.len() != b.items.len() { return false; }
                    return a.items.iter().zip(b.items.iter()).all(|(ai, bi)| {
                        Rc::ptr_eq(ai, bi)
                    });
                }
                false
            },
            DistanceFieldEnum::Coloring(a, i) => {
                if let DistanceFieldEnum::Coloring(b, i2) = other {
                    return Rc::<DistanceFieldEnum>::ptr_eq(&i,&i2) && a.eq(b);
                }
                false
            },
            DistanceFieldEnum::Subtract(a) => {
                if let DistanceFieldEnum::Subtract(b) = other {
                    return Rc::<DistanceFieldEnum>::ptr_eq(&a.left,&b.left) && 
                        Rc::<DistanceFieldEnum>::ptr_eq(&a.subtract,&b.subtract);
                }
                false
                
            },
            
            DistanceFieldEnum::Empty => {
                if let DistanceFieldEnum::Empty = other {
                    return true;
                }
                return false;
                
            },
        }
    }

    pub fn count_primitives(&self) -> u32 {
        match self {
            DistanceFieldEnum::Primitive(_) => 1,
            DistanceFieldEnum::Coloring(_, inner) => inner.count_primitives(),
            DistanceFieldEnum::Add(add) => add.items.iter().map(|i| i.count_primitives()).sum(),
            DistanceFieldEnum::Subtract(sub) => sub.left.count_primitives(),
            DistanceFieldEnum::Empty => 0,
        }
    }

    /// Count primitives but stop early once the count exceeds `limit`.
    /// Returns the actual count if <= limit, or limit + 1 to indicate "more than limit".
    pub fn count_primitives_up_to(&self, limit: u32) -> u32 {
        self.count_primitives_bounded(limit, 0)
    }

    fn count_primitives_bounded(&self, limit: u32, acc: u32) -> u32 {
        if acc > limit {
            return acc;
        }
        match self {
            DistanceFieldEnum::Primitive(_) => acc + 1,
            DistanceFieldEnum::Coloring(_, inner) => inner.count_primitives_bounded(limit, acc),
            DistanceFieldEnum::Add(add) => {
                let mut count = acc;
                for item in &add.items {
                    count = item.count_primitives_bounded(limit, count);
                    if count > limit { return count; }
                }
                count
            }
            DistanceFieldEnum::Subtract(sub) => sub.left.count_primitives_bounded(limit, acc),
            DistanceFieldEnum::Empty => acc,
        }
    }

    pub fn first_add(&self) -> Option<Add> {
        match self{
            DistanceFieldEnum::Primitive(_) => None,
            DistanceFieldEnum::Coloring(_, inner) => inner.first_add(),
            DistanceFieldEnum::Add(add) => Some(add.clone()),
            DistanceFieldEnum::Subtract(sub) => sub.left.first_add(),
            DistanceFieldEnum::Empty => None,
        }
    }

    fn print_graph_rec (&self, n : i32, f : &mut fmt::Formatter) -> fmt::Result{
        for _i in 0..n {
            print!(" ");
        }
        match self{
            DistanceFieldEnum::Primitive(_p) => {f.write_str("primitive\n")},
            DistanceFieldEnum::Coloring(_, inner) => 
                f.write_str("color\n").and(inner.as_ref().print_graph_rec(n + 1, f)),
            DistanceFieldEnum::Add(add) => {
                let mut result = f.write_str("add\n");
                for item in &add.items {
                    result = result.and(item.as_ref().print_graph_rec(n + 1, f));
                }
                result
            },
            DistanceFieldEnum::Subtract(sub) =>
            {
                f.write_str("subtract")?;
                sub.left.as_ref().print_graph_rec(n + 1, f)?;
                sub.subtract.as_ref().print_graph_rec(n + 1, f)
                },
            DistanceFieldEnum::Empty => fmt::Result::Ok(()),
        }
    }

    pub fn print_graph(&self, f: &mut fmt::Formatter) -> fmt::Result  {
        self.print_graph_rec(0, f)
    }
}

pub struct SdfPrinter{
    pub sdf : DistanceFieldEnum
}

impl Display for SdfPrinter{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.sdf.print_graph(f)
    }
}

impl PartialEq for DistanceFieldEnum{
    fn eq(&self, other: &Self) -> bool {
        match self {
            DistanceFieldEnum::Primitive(a) => {
                if let DistanceFieldEnum::Primitive(b) = other {
                    return a.eq(b);
                }
                return false;
            },
            DistanceFieldEnum::Coloring(a, ac) => {
                if let DistanceFieldEnum::Coloring(b, bc) = other {
                    return a.eq(b) && ac.eq(bc);
                }
                return false;
            },
            DistanceFieldEnum::Add(add) => {
                if let DistanceFieldEnum::Add(add2) = other {
                    return add.items.len() == add2.items.len()
                        && add.items.iter().zip(add2.items.iter()).all(|(a, b)| a.eq(b));
                }
                return false;
            },
            DistanceFieldEnum::Subtract(a) => {
                if let DistanceFieldEnum::Subtract(b) = other {
                    return a.eq(b);
                }
                return false;
            },
            DistanceFieldEnum::Empty => {
                if let DistanceFieldEnum::Empty = other {
                    return true;
                }
                return false;
            },
        }
    }
    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Eq for DistanceFieldEnum {
    
}

pub fn build_test() -> DistanceFieldEnum {
    let aabb2 = Sphere::new(Vec3::new(20.0, -100.0, 0.0), 80.0);
    let grad = Noise::new(
        1543,
        Color::WHITE,
        Color::ORANGE,
        aabb2.into(),
    );

    let sphere = Sphere::new(Vec3::new(-20.0, 0.0, 0.0), 2.0);

    let noise = Noise::new(
        123,
        Color::ORCHID,
         Color::WHITE,
        sphere.into(),
    );

    let sphere2 = Sphere::new(Vec3::new(0.0, -200.0, 0.0), 10.0);
    //let sphere3 = Sphere::new(Vec3::new(0.0, -200.0, 0.0), 180.0);
    //let shell = Subtract::new(sphere2.clone(), sphere3.clone(), 0.0);
    let noise2 = Noise::new(
        123,
        Color::OLIVE,
        Color::GRAY,
        sphere2.into(),
    );

    let sdf = DistanceFieldEnum::Empty {}
        .insert_2(noise)
        .insert_2(grad)
        .insert_2(noise2);
    return sdf;

    
}

pub fn build_test_solid() -> DistanceFieldEnum {
    let aabb2 : DistanceFieldEnum = Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0).into();
    let aabb2 = aabb2.colored(Color::WHITE);
    

    let sphere : DistanceFieldEnum = Sphere::new(Vec3::new(-2.0, 0.0, 0.0), 2.0).into();//color(Rgba([255, 0, 0, 255]));
    let sphere = sphere.colored(Color::RED);

    let sphere2 : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, -200.0, 0.0), 190.0).into();
    let sphere2 = sphere2.colored(Color::RED);
    

    let sdf = DistanceFieldEnum::Empty {}
        .insert_2(aabb2)
        .insert_2(sphere)
        .insert_2(sphere2);
    return sdf;

    
}


pub fn build_test2() -> DistanceFieldEnum {
    let mut sdf  = DistanceFieldEnum::Empty;
    let mut rng: rand::rngs::ThreadRng = thread_rng();
    //let v = (0..3).shuffle()
    let mut v = [0,1,2];
    v.shuffle(&mut rng);
    for x in v.iter(){
        let mut v = [0,1,2];
        v.shuffle(&mut rng);
    
        for y in v.iter() {
            let mut v = [0,1,2];
            v.shuffle(&mut rng);
    
            for z in v.iter() {
                let sphere = Sphere::new(Vec3::new(*x as f32* 5.0, *y as f32* 5.0, *z as f32 * 5.0), 2.0).color(Color::RED);
                sdf = sdf.insert_2(sphere);
            }
        }
    }
    
    return sdf;
}

pub fn build_test3() -> DistanceFieldEnum {
    let mut sdf  = DistanceFieldEnum::Empty;
    sdf = sdf.insert_2(Sphere::new(Vec3::new(0.0,0.0,0.0), 2.0).color(Color::RED));
    sdf = sdf.insert_2(Sphere::new(Vec3::new(25.0,0.0,0.0), 2.0).color(Color::RED));
    //sdf = sdf.insert_2(Sphere::new(Vec3f::new(1.0,0.0,0.0), 2.0).color(Rgba([255, 0, 0, 255])));
    //sdf = sdf.insert_2(Sphere::new(Vec3f::new(6.0,0.0,0.0), 2.0).color(Rgba([255, 0, 0, 255])));
    
    return sdf;
}

pub fn build_big(n : i32) -> DistanceFieldEnum {
    let mut sdf  = DistanceFieldEnum::Empty;
    //let mut rng: rand::rngs::ThreadRng = thread_rng();
    //let mut bounds_cache = HashMap::new();
    
    for x in 0..n{
        for y in 0..n {
            for z in 0..n {
                //println!("{} {} {}", x, y, z);
                let sphere = Sphere::new(Vec3::new(x as f32* 10.0, y as f32* 10.0, z as f32 * 10.0), 3.0);
                let sphere2 = sphere.clone();
                //sdf = sdf.insert_4(sphere, &sphere2, &mut bounds_cache);
                sdf = sdf.insert_3(sphere, &sphere2);
            }
            sdf = sdf.optimize_bounds();
        }
    }
    sdf = sdf.colored(Color::CYAN);
    return sdf;
}


impl fmt::Display for DistanceFieldEnum{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DistanceFieldEnum::Primitive(primitive) => {
                match primitive {
                    Primitive::Sphere(sphere) => write!(f, "(Sphere {} {})", sphere.center, sphere.radius),
                    Primitive::Aabb(aabb) => write!(f, "(AABB {} {})", aabb.center, aabb.radius),
                }
            },
            DistanceFieldEnum::Add(add) => {
                write!(f, "(")?;
                for (i, item) in add.items.iter().enumerate() {
                    if i > 0 { write!(f, "\n ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, "\n)")
            },
            DistanceFieldEnum::Coloring(c, inner) => {
                match c {
                    Coloring::SolidColor(c) => write!(f, "(color: [{}] {})", c, inner),
                    Coloring::Gradient(_g) => write!(f, "(gradient: ? {})", inner),
                    Coloring::Noise(n) => write!(f, "(noise: (n: {}) {})", n, inner),
                    Coloring::ColorScale(s) => write!(f, "(colorscale: [{} {} {}] {})", s.scale.x, s.scale.y, s.scale.z, inner)
                }
            },
            DistanceFieldEnum::Subtract(sub) => write!(f, "(subtract {} {})", sub.left, sub.subtract),
            DistanceFieldEnum::Empty => write!(f, "[empty]"),
        }
    }
}


#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use nalgebra::ComplexField;
    use rand::{rngs::{ThreadRng, StdRng}, SeedableRng};

    use super::*;


    #[test]
    fn test_optimize_bounds() {
        let sdf = build_test();

        let opt = sdf.optimize_bounds();
        
        let pt = Vec3::new(0.0, 0.0, 0.0);

        let a = sdf.distance(pt);
        let b = opt.distance(pt);
        assert_eq!(a, b);
        println!("Optimal!");
        println!("{:?}", opt);
    }

    #[test]
    fn test_optimal_ray() {
        let sdf = build_test();

    }

    #[test]
    fn test_sdf_format(){
        let sdf = build_test2();
        println!("{}", sdf);
    }

    #[test]
    fn test_build_big_optimize(){
        let sdf = build_big(6);
        let sdf2 = sdf.optimized_for_block(Vec3::new(0.0, 0.0, 0.0), 5.0);
        println!("sdf2: {}", sdf2);
    }
    #[test]
    fn test_optimized_bounds(){
        let sdf = DistanceFieldEnum::Empty
            .insert_2(Sphere::new(Vec3::new(0.0,0.0,0.0), 1.0))
            .insert_2(Sphere::new(Vec3::new(2.0,0.0,0.0), 1.0));
        
        let sdfopt = sdf.optimize_bounds();
        println!("{}", sdf);
        println!("{}", sdfopt);
        
        let mut rng = StdRng::seed_from_u64(2);

        let d1 = Vec3::new(1.0, 5.5, 0.0);
        let a1 = sdf.distance(d1);
        let c1 = sdfopt.distance(d1);
        let error = (a1 - c1) / a1;
        println!("{} {} {}", a1, c1 ,error);
        
        
        for _ in 0..1000 {
             
            let d = Vec3::new(rng.gen_range(-10.0..10.0), rng.gen_range(-10.0..10.0),rng.gen_range(-10.0..10.0));
            
            
            let a = sdf.distance(d);
            let c = sdfopt.distance(d);
            
            println!("bounds??: {} {} {} {}", d, a, c, (a - c) / a);
            assert!(c <= a);

            let error = (a - c) / a;
            if a < 0.0 {
                assert!((a - c).abs() < 0.000001);
            }else{
                assert!(error < 0.4);
            }   
        }
    }
    //#[test]
    fn test_build_big_optimize2(){
        let sdf = build_big(4);
        let sdfopt = sdf.optimize_bounds();
        
        let bounds =  sdf.calculate_sphere_bounds();
        let mut rng = thread_rng();
        for _ in 0..10 {
            let size = rng.gen_range(1.0..10.0);
            let halfsize = size * 0.5;
            let d = Vec3::new(rng.gen_range(0.0..12.0), rng.gen_range(0.0..12.0),rng.gen_range(0.0..12.0));
            let d2: Vec3 = Vec3::new(rng.gen_range(-halfsize..halfsize)
                , rng.gen_range(-halfsize..halfsize)
                , rng.gen_range(-halfsize..halfsize));
            let block = sdf.optimized_for_block(d + d2, size);
            
            let a = sdf.distance(d);
            let b = block.distance(d);
            let c = sdfopt.distance(d);
            println!("OPTIMIZED:");
            println!("{} unoptimized: {}", block, sdf);
            println!("block: {} {} {} {} {}", d, a, b, size, (a - c).abs());
            assert_eq!(a, b);
            if a < 0.0 {
                assert!((a - c).abs() < 0.000001);
            }else{
                if (a - c).abs() > a * 0.01{
                    println!("{} {}", a, c);
                }
                assert!((a - c).abs() < a * 0.01);

            }
            
        }
    }

    fn run_test_on_range(ground_truth : DistanceFieldEnum, 
        optimized_sdf : DistanceFieldEnum, 
        center : Vec3, size: Vec3, n : i32, e : f32){
        let radius = size * 0.5;
        let mut rng = StdRng::seed_from_u64(2);

        for i in 0..n {
            let offset = radius.map(|x| rng.gen_range(-x..x));
            let pos = center + offset;
            let a = ground_truth.distance(pos);
            let b = optimized_sdf.distance(pos);
            println!("{} vs {}", (b - a) / a,  e);
            assert!((b - a) / a < e);
        }
    }

    #[test]
    fn test_optimize_subtract(){
        let sdf = DistanceFieldEnum::Empty
            .insert_2(Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0).color(Color::RED))
            .insert_2(Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0).color(Color::BLUE));
        let sdf : DistanceFieldEnum = Subtract::new(sdf, Sphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0), 0.1).into();

        let opt = sdf.optimize_bounds().optimize_bounds();
        println!("Opt: {}", opt);
        run_test_on_range(sdf.clone(), opt, Vec3::new(1.0,0.0,0.0), Vec3::new(5.0, 5.0, 5.0), 1000, 0.2);
    
        let opt2 = sdf.optimized_for_block(Vec3::new(0.0,0.0,0.0), 1.0);
        println!("Opt2: {}", opt2);

    
    }

    

    #[test]
    fn test_sdf_hashing(){
        let mut map = HashSet::new();
        let sdf = DistanceFieldEnum::Empty
            .insert_2(Sphere::new(Vec3::new(1.0, 0.0, 5.0), 1.0))
            .insert_2(Sphere::new(Vec3::new(2.0, 1.0, 2.0), 1.0));
        let sdf2 = DistanceFieldEnum::Empty
            .insert_2(Sphere::new(Vec3::new(1.0, 0.0, 5.0), 1.0))
            .insert_2(Sphere::new(Vec3::new(3.0, 1.0, 2.0), 1.0));
        let sdf4 = DistanceFieldEnum::Empty
            .insert_2(Sphere::new(Vec3::new(1.0, 0.0, 5.0), 1.0))
            .insert_2(Sphere::new(Vec3::new(3.0, 1.0, 2.0), 1.0));
        
        sdf.build_map(&mut map);
        sdf2.build_map(&mut map);

        let sdf3 = map.get(&sdf4).unwrap();
        &sdf.clone().build_map(&mut map);

        println!(">> {}", map.len())
        
    }
    #[test]
    fn test_super_gradient2(){
        let sdf = build_big(3);
        
        //let sdf = sdf.insert_2(Sphere::new(Vec3::new(31.0,0.0,0.0), 1.0));
        println!("{}", sdf.size());
        println!("{}", SdfPrinter{sdf: sdf.clone()});
        
        if let Some(a) = sdf.first_add() {
            println!("Balanced? {}", a.items.iter().map(|i| i.size().to_string()).collect::<Vec<_>>().join(", "));
        }
        let pos = Vec3::new(10.0, 10.0, 0.0);
        println!("optimizing:");
        let focus = sdf.distance_and_optimize(pos, 0.001, &mut HashSet::new()).1;
        
        let g = sdf.gradient(pos, 0.001);
        println!("focus: {} {}", g, focus);
        
    }


    #[test]
    fn test_super_gradient(){
        let s1 : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 10.0).into();
        let s2 : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 50.0, 0.0), 10.0).into();
        let a1 : DistanceFieldEnum = Add::new(s1,s2).into();
        
        let g = a1.gradient(Vec3::new(-20.0, 50.0, 0.0), 0.1);
        fn f32_eq(a : f32, b: f32) -> bool {
            (a - b).abs() < 0.0001
        }
        fn vec3_eq(a : Vec3, b : Vec3) -> bool {
            f32_eq(a.x, b.x) && f32_eq(a.y, b.y) && f32_eq(a.z, b.z)
        }
        assert!(f32_eq(1.0, g.length()));
        assert!(vec3_eq(g, Vec3::new(-1.0, 0.0, 0.0)));

        let g = a1.gradient(Vec3::new(20.0, 50.0, 0.0), 0.1);
        assert!(vec3_eq(g, Vec3::new(1.0, 0.0, 0.0)));
        let g = a1.gradient(Vec3::new(0.0, 50.0, 20.0), 0.1);
        assert!(vec3_eq(g, Vec3::new(0.0, 0.0, 1.0)));
        
    }

    #[test]
    fn test_optimize_preserve_colors() {
        let sdf = build_test().optimize_bounds();
        let sdf2 = sdf.optimized_for_block(Vec3::new(-2.0, 0.0, 0.0), 5.0);
        println!("g: {}", sdf);
    }

    #[test]
    fn optimize_bounds_bug1(){
        //(subtract ((color: [(0 0 1 1)] (Sphere (0, 50, 50) 50))
 //(color: [(1 0 0 1)] (Sphere (0, 55, 0) 10))
// ) (Sphere (9.092156, 59.161907, 0.119892195) 2))

    let sphere1 = DistanceFieldEnum::sphere(Vec3::new(0.0, 50.0, 50.0), 50.0);
    let sphere2 = DistanceFieldEnum::sphere(Vec3::new(0.0, 55.0, 0.0), 10.0);
    let subcenter = Vec3::new(9.092156, 59.161907, 0.119892195);
    let subsphere = DistanceFieldEnum::sphere(subcenter, 2.0);
    
    let sdf = sphere1.add(sphere2).subtract(subsphere);
    let d1 = sdf.distance(subcenter);
    let sdf2 = sdf.optimize_bounds();
    let d2 = sdf2.distance(subcenter);

    println!("sdf2: {}   {}",  d1, d2);
    assert!((d1 - d2) < 0.001);

    }

    #[test]
    fn test_distance_bug1(){
        let s1 : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 50.0, 50.0), 50.0).into();
        let s2 : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 55.0, 00.0), 10.0).into();
        let a1 : DistanceFieldEnum = Add::new(s1,s2).into();
        let sub = Vec3::new(12.840058, 74.62816, 8.423447);
        let a1 = a1.subtract(DistanceFieldEnum::sphere(sub, 2.0)).optimize_bounds();
        let d = a1.distance(Vec3::new(11.473644, 60.98243, 19.648586));
        println!("?? {}  {}", d, a1);
        assert!(!d.is_nan());

 
    }


}
