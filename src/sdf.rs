
use image::{Rgba};
use noise::{NoiseFn, Perlin};
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashSet, HashMap};
use std::f32::consts::SQRT_2;
use std::fmt::Display;
use std::{fmt, io};
use std::hash::{Hash, Hasher};
//, Simplex, SuperSimplex};
use std::rc::Rc;

use crate::color::Color;
use crate::vec3::Vec3;

const SQRT3: f32 = 1.73205080757;

const BIGPRIME : i64 = 1844674407370955155;

    pub fn mix(a: i32, b: i32) -> i32{
        let a = a as i64;
        let b = b as i64;
        let mut x = BIGPRIME;
        for _ in 0..5 {
            x = x.wrapping_mul(BIGPRIME).wrapping_add(a);
            x = x.wrapping_mul(BIGPRIME).wrapping_add(b);
        }
        x as i32
    }

    trait Hash2 {
        fn hash2(&self) -> i32;
    }

    impl Hash2 for f32 {
        fn hash2(&self) -> i32 {
            (f32::to_bits(*self) as i64).wrapping_mul(BIGPRIME) as i32
        }
    }

    impl Hash2 for Vec3{
        fn hash2(&self) -> i32 {
            mix(mix(self.x.hash2(), self.y.hash2()), self.z.hash2())
        }
    }

#[derive(Clone, Debug, PartialEq)]
pub enum Primitive{
    Sphere(Sphere),
    Aabb(Aabb)
}

impl Hash for Primitive {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl Primitive{
    fn distance(&self, p : Vec3) -> f32 {
        
        match self{
            Primitive::Sphere(s) => s.distance(p),
            Primitive::Aabb(s) => s.distance(p),
        }
    }

    fn hash(&self) -> i32 {
        match self {
            Primitive::Sphere(s) => mix(s.center.hash2(),s.radius.hash2()),
            Primitive::Aabb(s) => mix(s.center.hash2(), s.radius.hash2()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Coloring{
    SolidColor(Color),
    Gradient(Gradient),
    Noise(Noise)
}
impl Hash2 for Rgba<u8> {
    fn hash2(&self) -> i32 {
        mix(mix(self[0] as i32, self[1] as i32), mix(self[2] as i32, self[3] as i32))
    }
}

impl Hash2 for Color {
    fn hash2(&self) -> i32 {
        self.to_u8_rgba().hash2()
    }
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
        }
    }

    pub fn hash(&self) -> i32 {
        match self {
            Coloring::SolidColor(v) => v.hash2(),
            Coloring::Gradient(v) => mix(mix(v.c1.hash2(), v.c2.hash2()), mix(v.p1.hash2(), v.p2.hash2())),
            Coloring::Noise(v) => mix(mix(v.c1.hash2(), v.c2.hash2()), v.seed as i32)
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
    fn distance(&self, pos: Vec3) -> f32 {
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

#[derive(Clone, PartialEq, Debug, Hash)]
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
    noise: Rc<Perlin>,
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
            noise: Rc::new(Perlin::new(seed)),
            c1: c1,
            c2: c2
        }), Rc::new(inner))
    }

    fn color(&self, pos: Vec3) -> Color {
        let pos2 = pos * 0.25;
        let pos3 = pos * 4.0;
        let n1 = self
            .noise
            .get([pos.x as f64, pos.y as f64, pos.z as f64]);
        let n2 = self
            .noise
            .get([pos2.x as f64, pos2.y as f64, pos2.z as f64]);
        let n3 = self
            .noise
            .get([pos3.x as f64, pos3.y as f64, pos3.z as f64]);
        let color = rgba_interp(self.c1, self.c2, 0.5 * (n1 + n2 + n3) as f32);
        //if color[3] < 255 {
        //    let mut colorbase = self.inner.color(pos);
        //    colorbase.blend(&color);
        //    return colorbase;
        //}
        return color;
    }
}


#[derive(Clone, PartialEq, Debug)]
pub struct Add {
    pub(crate) left: Rc<DistanceFieldEnum>,
    pub(crate) right: Rc<DistanceFieldEnum>,
    size : u32,
    pub(crate) bounds: Sphere,
    hash: u64
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
        Add::from_rc(Rc::new(left), Rc::new(right))
    }

    pub fn new2(left: DistanceFieldEnum, right: DistanceFieldEnum) -> DistanceFieldEnum {
        DistanceFieldEnum::Add(Add::from_rc(Rc::new(left), Rc::new(right)))
    }

    fn from_rc(left: Rc<DistanceFieldEnum> , right: Rc<DistanceFieldEnum>) -> Add {
        let size = left.size() + right.size() + 1;

        let s1 = right.calculate_sphere_bounds();
        let s2 = left.calculate_sphere_bounds();
        let mut hasher = DefaultHasher::new();
        let num = 32143232;
        
        num.hash(&mut hasher);
        left.hash(&mut hasher);
        right.hash(&mut hasher);
        let hash = hasher.finish();
        Add {
            left: left,
            right: right,
            size: size,
            bounds: Sphere::two_sphere_bounds(&s1, &s2),
            hash: hash
        }
    }

    fn distance(&self, pos: Vec3) -> f32 {
        let d1 = self.bounds.distance(pos);
        if d1 > self.bounds.radius {
            return d1;
        }

        f32::min(self.left.distance(pos), self.right.distance(pos))
    }

    fn color(&self, p: Vec3) -> Color {
        let ld = self.left.distance(p);
        let rd = self.right.distance(p);
        if ld < rd {
            return self.left.color(p);
        }
        return self.right.color(p);
        
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
    
    
    pub fn topology_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash_topology(&mut hasher);
        hasher.finish()
    }

    fn hash_topology<H: Hasher>(&self, state: &mut H) {
        match self {
            DistanceFieldEnum::Empty => 0u8.hash(state),
            DistanceFieldEnum::Primitive(p) => {
                1u8.hash(state);
                core::mem::discriminant(p).hash(state);
            }
            DistanceFieldEnum::Coloring(c, inner) => {
                2u8.hash(state);
                core::mem::discriminant(c).hash(state);
                inner.hash_topology(state);
            }
            DistanceFieldEnum::Add(add) => {
                3u8.hash(state);
                // Order-invariant: hash children in sorted order by their topology hash
                let mut left_h = DefaultHasher::new();
                add.left.hash_topology(&mut left_h);
                let lh = left_h.finish();
                let mut right_h = DefaultHasher::new();
                add.right.hash_topology(&mut right_h);
                let rh = right_h.finish();
                let (lo, hi) = if lh <= rh { (lh, rh) } else { (rh, lh) };
                lo.hash(state);
                hi.hash(state);
            }
            DistanceFieldEnum::Subtract(sub) => {
                4u8.hash(state);
                sub.left.hash_topology(state);
                sub.subtract.hash_topology(state);
            }
        }
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
            if d < 0.001 {
                return Some((total_distance, mpos));
            }
        }
        None
    }

    pub fn optimize_add(add: &Add, block_center: Vec3, size: f32, cache: &mut HashSet<DistanceFieldEnum>, min_d : f32) -> Rc<DistanceFieldEnum> {
        let d1 = add.left.distance(block_center);
        let d2 = add.right.distance(block_center);
        let new_min_d = (f32::min(d1, f32::min(d2, min_d)) * 10.0).floor() / 10.0;
        let left_opt = add.left.optimized_for_block2(block_center, size, cache, new_min_d);
        let right_opt = add.right.optimized_for_block2(block_center, size, cache, new_min_d);

        if matches!(left_opt.as_ref(), DistanceFieldEnum::Empty) {
            return right_opt;
        }
        if matches!(right_opt.as_ref(), DistanceFieldEnum::Empty) {
            return left_opt;
        }
        
        let left_d = left_opt.distance(block_center);
        let right_d = right_opt.distance(block_center);
        let half = size / 2.0;
        if left_d > right_d + half * SQRT3 * SQRT_2{
            return right_opt;
        }
        if right_d > left_d + half * SQRT3 * SQRT_2{
            return left_opt;
        }
        if f32::min(right_d, left_d) > min_d + half * SQRT3 * 2.0  {
            return DistanceFieldEnum::Empty{}.into();
        }
        if f32::min(right_d, left_d) > half * SQRT3 * 2.0 * 1.5  {
            if right_d < left_d {
                return right_opt;
            }
            return left_opt;
        }
        
        if left_opt.eq(&add.left) && right_opt.eq(&add.right) {
            let add2 = add.clone();
            return Rc::new(DistanceFieldEnum::Add(add2).into());
        }
        return Rc::new(Add::from_rc(left_opt, right_opt).into());
    }
    pub fn optimized_for_block(&self, block_center: Vec3, size: f32, cache: &mut HashSet<DistanceFieldEnum>) -> Rc<DistanceFieldEnum> {
        return self.optimized_for_block2(block_center, size, cache, f32::INFINITY);
    }
    
    pub fn optimized_for_block2(&self, block_center: Vec3, size: f32, cache: &mut HashSet<DistanceFieldEnum>, min_d :f32) -> Rc<DistanceFieldEnum> {
        match self {
            DistanceFieldEnum::Add(add) => {
                DistanceFieldEnum::optimize_add(add, block_center,size, cache, min_d)
            },
            DistanceFieldEnum::Subtract(sub) => {
                let optsub = sub.subtract.optimized_for_block2(block_center, size, cache, min_d);
                
                let subtract_d = optsub.distance(block_center);
                let left2 = sub.left.optimized_for_block2(block_center, size, cache, min_d);
                if subtract_d > size / 2.0 * SQRT3 * 2.0{
                    if left2.eq(&sub.left) {
                        return sub.left.clone();
                    }
                    return left2;
                }
               
                if left2.eq(&sub.left) && optsub.eq(&sub.subtract) {
                    return Rc::new(self.clone());
                }else if left2.eq(&sub.left){
                    return DistanceFieldEnum::Subtract( Subtract {left : sub.left.clone(),
                        subtract: optsub, k: sub.k}).into()
                }else if optsub.eq(&sub.subtract){
                    return DistanceFieldEnum::Subtract( Subtract {left : left2,
                        subtract: sub.subtract.clone(), k: sub.k}).into()
                }
                
                return DistanceFieldEnum::Subtract( Subtract {left : left2,
                    subtract: optsub, k: sub.k}).into()
               
            },
            DistanceFieldEnum::Coloring(c, i) => {
                let opt = i.optimized_for_block2( block_center, size, cache, min_d);
                if matches!(opt.as_ref(), DistanceFieldEnum::Empty) {
                    return Rc::new(DistanceFieldEnum::Empty);
                }
                if opt.eq(i) {
                    return Rc::new(self.clone());
                }
                return Rc::new(DistanceFieldEnum::Coloring(c.clone(), opt).into());
            }
            
            _ =>{
                let sb = self.calculate_sphere_bounds();
                if !sb.overlaps_aabb(block_center, size / 2.0) && min_d.is_finite() {
                    return Rc::new(DistanceFieldEnum::Empty);
                }
                if self.distance(block_center) > min_d + size / 2.0 * SQRT3 * 2.0 {
                    return Rc::new(DistanceFieldEnum::Empty);
                }
                return Rc::new(self.clone())
            }
        }
    }

    pub fn distance_and_optimize(&self, pos: Vec3, size: f32, cache: &mut HashSet<DistanceFieldEnum>) -> (f32, DistanceFieldEnum) {
        let pos2 : Vec3 = pos.map(|x| f32::floor(x / size) * size);
        let sdf2 =
            self.optimized_for_block(pos2 + Vec3::new(size * 0.5, size * 0.5, size * 0.5), size, cache);//.cached(cache);
        
        return (sdf2.distance(pos), sdf2.as_ref().clone());
    }

    pub fn insert(&self, sdf: DistanceFieldEnum) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Empty => sdf,
            _ => Add::new(self.clone(), sdf).into(),
        }
    }

    pub fn insert_3<T: Into<DistanceFieldEnum>>(&self, sdf: T, bounds: &Sphere) -> DistanceFieldEnum {
        
        match self {
            DistanceFieldEnum::Add(add) => {
                let l0 = add.left.calculate_sphere_bounds().center;
                let l1 = add.right.calculate_sphere_bounds().center;
                let l3 = (l0 - l1).length();
                
                let s1 = add.left.size();
                let s2 = add.right.size();
                
                let bias = s1 as f32 / (s2 + s1) as f32;

                let d1 = add.left.distance(bounds.center) * bias;
                let d2 = add.right.distance(bounds.center) * (1.0 - bias);
                
                //println!("Into add! {} {} {}", d1, d2, l3);
                
                if d1 < d2 && d1 < l3 * 0.9 {
                    return Add::from_rc(Rc::new(add.left.insert_3(sdf, bounds)),add.right.clone()).into();
                }else if d2 < d1 && d2 < l3* 0.9 {
                    return Add::from_rc( add.left.clone(), Rc::new(add.right.insert_3(sdf, bounds))).into();
                }
                return self.insert(sdf.into());
            },
            //DistanceFieldEnum::Coloring(c, other) => DistanceFieldEnum::Coloring(c.clone(), Rc::new(other.insert_2(sdf))),
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
                let left = add.left.calculate_aabb_bounds();
                let right = add.right.calculate_aabb_bounds();
                left.union(&right)
            }
            DistanceFieldEnum::Subtract(sub) => sub.left.calculate_aabb_bounds(),
        }
    }

    pub fn calculate_sphere_bounds2(&self, cache: &mut HashMap<DistanceFieldEnum, Sphere>) -> Sphere {
        if let Some(x) = cache.get(self) {
            return x.clone();
        }
        let r = 
        match self {
            DistanceFieldEnum::Primitive(p) =>{
                match p {
                    Primitive::Sphere(s) => s.clone(),
                    Primitive::Aabb(aabb) => Sphere::new(aabb.center, aabb.radius.length()),
                }
            },
            DistanceFieldEnum::Empty => Sphere::new(Vec3::zeros(), f32::INFINITY),
            DistanceFieldEnum::Coloring(_,inner) => inner.calculate_sphere_bounds2(cache),
            DistanceFieldEnum::Add(add) => {
                let left = add.left.calculate_sphere_bounds2(cache);
                let right = add.right.calculate_sphere_bounds2(cache);

                Sphere::two_sphere_bounds(&left, &right)
            },
            DistanceFieldEnum::Subtract(sub) => sub.left.calculate_sphere_bounds2(cache)
        };
        cache.insert(self.clone(), r.clone());
        r
    }

    pub fn optimize_bounds(&self) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Add(add) => {
                add.clone().into()
            },
            DistanceFieldEnum::Subtract(sub) => {
                let left2 = sub.left.optimize_bounds();
                let subbounds = sub.subtract.calculate_sphere_bounds();

                // If the left side is an Add, try to push the subtraction into
                // only the branch(es) whose bounds overlap the subtracted volume.
                if let DistanceFieldEnum::Add(inner_add) = &left2 {
                    let left_bounds = inner_add.left.calculate_sphere_bounds();
                    let right_bounds = inner_add.right.calculate_sphere_bounds();
                    let left_overlaps = left_bounds.overlaps(&subbounds);
                    let right_overlaps = right_bounds.overlaps(&subbounds);

                    if !left_overlaps && !right_overlaps {
                        // neither bounds overlaps the subtraction -> subtraction has no effect
                        return DistanceFieldEnum::Add(inner_add.clone());
                    } else if !left_overlaps {
                        // left does not overlap -> move the subtraction into the right only
                        return Add::new(
                            inner_add.left.as_ref().clone(),
                            Subtract::new(inner_add.right.as_ref().clone(), sub.subtract.as_ref().clone(), sub.k).into()
                        ).into();
                    } else if !right_overlaps {
                        // right does not overlap -> move the subtraction into the left only
                        return Add::new(
                            Subtract::new(inner_add.left.as_ref().clone(), sub.subtract.as_ref().clone(), sub.k).into(),
                            inner_add.right.as_ref().clone()
                        ).into();
                    }
                    // both overlap -> fall through to keep the subtraction on the whole Add
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
            DistanceFieldEnum::Coloring(c, inner) => c.color(pos),
            DistanceFieldEnum::Subtract(sub) => sub.left.color(pos)
        }
    }

    pub fn gradient(&self, pos: Vec3, size: f32) -> Vec3 {
        let focus = self.optimized_for_block(pos, size, &mut HashSet::new());
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
                    return
                        (Rc::<DistanceFieldEnum>::ptr_eq(&a.left, &b.left)
                         && Rc::<DistanceFieldEnum>::ptr_eq(&a.right, &b.right))
                        || (a.left.equals(&b.left) &&  a.right.equals(&b.right));
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
                    return Rc::<DistanceFieldEnum>::ptr_eq(&a.left, &b.left)
                    &&Rc::<DistanceFieldEnum>::ptr_eq(&a.right, &b.right);
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

    pub fn build_map(&self, m : &mut HashSet<DistanceFieldEnum>){

        if m.insert(self.clone()){
            match self {
                DistanceFieldEnum::Primitive(_) => {},
                DistanceFieldEnum::Coloring(_, inner) => inner.build_map(m),
                DistanceFieldEnum::Add(add) => {
                    add.left.build_map(m);
                    add.right.build_map(m);
                },
                DistanceFieldEnum::Subtract(sub) => {
                    sub.left.build_map(m);
                    sub.subtract.build_map(m);
                },
                DistanceFieldEnum::Empty => {},
            }
        }
    }
    pub fn cached(&self, m : &mut HashSet<DistanceFieldEnum>) -> DistanceFieldEnum{

        if m.insert(self.clone()){
            match self {
                DistanceFieldEnum::Primitive(_) => {},
                DistanceFieldEnum::Coloring(_, inner) => inner.build_map(m),
                DistanceFieldEnum::Add(add) => {
                    add.left.build_map(m);
                    add.right.build_map(m);
                },
                DistanceFieldEnum::Subtract(sub) => {
                    sub.left.build_map(m);
                    sub.subtract.build_map(m);
                },
                DistanceFieldEnum::Empty => {},
            }
        }
        return m.get(&self).unwrap().clone();
    }

    pub fn size(&self) -> u32{
        match self {
            DistanceFieldEnum::Primitive(_) => 1,
            DistanceFieldEnum::Coloring(_, inner) => inner.size(),
            DistanceFieldEnum::Add(add) => add.size,
            DistanceFieldEnum::Subtract(sub) => {
                sub.left.size() + sub.subtract.size()
            },
            DistanceFieldEnum::Empty => 1,
        }
    }
    pub fn count_primitives(&self) -> u32 {
        match self {
            DistanceFieldEnum::Primitive(_) => 1,
            DistanceFieldEnum::Coloring(_, inner) => inner.count_primitives(),
            DistanceFieldEnum::Add(add) => add.left.count_primitives() + add.right.count_primitives(),
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
                let left_count = add.left.count_primitives_bounded(limit, acc);
                if left_count > limit {
                    return left_count;
                }
                add.right.count_primitives_bounded(limit, left_count)
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
        for i in 0..n {
            print!(" ");
        }
        match self{
            DistanceFieldEnum::Primitive(p) => {f.write_str("primitive\n")},
            DistanceFieldEnum::Coloring(_, inner) => 
                f.write_str("color\n").and(inner.as_ref().print_graph_rec(n + 1, f)),
            DistanceFieldEnum::Add(add) => 
            
                f.write_str("add\n").and(add.left.as_ref().print_graph_rec(n + 1, f))
                    .and(add.right.as_ref().print_graph_rec(n + 1, f))
                ,
            DistanceFieldEnum::Subtract(sub) => 
            {
                f.write_str("subtract"); 
                sub.left.as_ref().print_graph_rec(n + 1, f);
                sub.subtract.as_ref().print_graph_rec(n + 1, f)
                },
            DistanceFieldEnum::Empty => fmt::Result::Ok(()),
        }
    }

    pub fn print_graph(&self, f: &mut fmt::Formatter) -> fmt::Result  {
        self.print_graph_rec(0, f)
    }
}

impl Hash for DistanceFieldEnum {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            DistanceFieldEnum::Add(add) => {
                add.hash.hash(state);
            },
            DistanceFieldEnum::Primitive(p) => {
                state.write_i32(p.hash());
            }
            DistanceFieldEnum::Coloring(c,i ) => {
                state.write_i64(323543265);
                state.write_i32(c.hash());
                i.hash(state);
            }
            DistanceFieldEnum::Subtract(sub) => {
              state.write_i64(33543265);
              sub.left.hash(state);
              sub.subtract.hash(state);
              state.write_i32(sub.k.hash2());   
            },
            DistanceFieldEnum::Empty => {
                321321532.hash(state);
            }
        }
        
    }
}

struct SdfPrinter{
    sdf : DistanceFieldEnum
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
                    return add2.hash == add.hash && add.left.eq(&add2.left) && add.right.eq(&add2.right);
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
            DistanceFieldEnum::Add(add) => write!(f, "({}\n {}\n)", add.left, add.right),
            DistanceFieldEnum::Coloring(c, inner) => {
                match c {
                    Coloring::SolidColor(c) => write!(f, "(color: [{}] {})", c, inner),
                    Coloring::Gradient(g) => write!(f, "(gradient: ? {})", inner),
                    Coloring::Noise(n) => write!(f, "(noise: (n: {}) {})", n, inner)
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
        let sdf2 = sdf.optimized_for_block(Vec3::new(0.0, 0.0, 0.0), 5.0, &mut HashSet::new());
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
            let block = sdf.optimized_for_block(d + d2, size, &mut HashSet::new());
            
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
    
        let opt2 = sdf.optimized_for_block(Vec3::new(0.0,0.0,0.0), 1.0, &mut HashSet::new());
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
            println!("Balanced? {} {}", a.left.size(),  a.right);
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
        let sdf2 = sdf.optimized_for_block(Vec3::new(-2.0, 0.0, 0.0), 5.0, &mut HashSet::new());
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
