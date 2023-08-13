
use image::{Rgba};
use noise::{NoiseFn, Perlin};
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashSet, HashMap};
use std::env::JoinPathsError;
use std::fmt::Display;
use std::{fmt, io};
use std::hash::{Hash, Hasher};
//, Simplex, SuperSimplex};
use std::rc::Rc;

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
    SolidColor(Rgba<u8>),
    Gradient(Gradient),
    Noise(Noise)
}
impl Hash2 for Rgba<u8> {
    fn hash2(&self) -> i32 {
        mix(mix(self[0] as i32, self[1] as i32), mix(self[2] as i32, self[3] as i32))
    }
}

impl Coloring {
    pub fn color(&self, p : Vec3) ->Rgba<u8> {
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
    center: Vec3,
    radius: f32
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32) -> Sphere {
        Sphere {
            center: center,
            radius: radius
        }
    }

    pub fn color(&self, color: Rgba<u8>) -> DistanceFieldEnum {
        DistanceFieldEnum::Coloring(Coloring::SolidColor(color),
        Rc::new(Sphere {
            center: self.center,
            radius: self.radius
        }.into()))
    }

    pub fn two_sphere_bounds(a: &Sphere, b: &Sphere) -> Sphere {
        let n2 = (a.center - b.center).length();
        if n2 <= 0.000001 {
            return match a.radius < b.radius{
                true => b,
                false => a
            }.clone();
        }
        let r2ld = (a.center - b.center) / n2;
        let leftext = a.center + r2ld * a.radius;
        let rightext = b.center - r2ld * b.radius;

        let center = (leftext + rightext) * 0.5;
        let radius = (leftext - rightext).length() * 0.5;

        return Sphere::new(center, radius);
    }

    pub fn overlaps(&self, other: &Sphere) -> bool {
        (self.center - other.center).length() - self.radius - other.radius < 0.0
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

#[derive(Clone, PartialEq, Debug, Hash)]
pub struct Aabb {
    radius: Vec3,
    center: Vec3
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
    p1: Vec3,
    p2: Vec3,
    c1: Rgba<u8>,
    c2: Rgba<u8>,
}

impl Gradient {
    pub fn new(
        p1: Vec3,
        p2: Vec3,
        c1: Rgba<u8>,
        c2: Rgba<u8>,
        inner: Rc<DistanceFieldEnum>,
    ) -> DistanceFieldEnum {
        DistanceFieldEnum::Coloring(Coloring::Gradient(Gradient {
            p1: p1,
            p2: p2,
            c1: c1,
            c2: c2,
        }),inner)
    }

    pub fn color(&self, p: Vec3) -> Rgba<u8> {
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
    seed: u32,
    c1: Rgba<u8>,
    c2: Rgba<u8>
}

impl PartialEq for Noise {
    fn eq(&self, other: &Self) -> bool {
        return self.seed == other.seed
            && self.c1 == other.c1
            && self.c2 == other.c2
    }
}

impl Noise {
    pub fn new(seed: u32, c1: Rgba<u8>, c2: Rgba<u8>, inner: DistanceFieldEnum) -> DistanceFieldEnum {
        DistanceFieldEnum::Coloring(Coloring::Noise(Noise {
            seed: seed,
            noise: Rc::new(Perlin::new(seed)),
            c1: c1,
            c2: c2
        }), Rc::new(inner))
    }

    fn color(&self, pos: Vec3) -> Rgba<u8> {
        let pos1 = pos * 1.0;
        let pos2 = pos1 * 0.25;
        let pos3 = pos1 * 4.0;
        let n1 = self
            .noise
            .get([pos1.x as f64, pos1.y as f64, pos1.z as f64]);
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
    left: Rc<DistanceFieldEnum>,
    right: Rc<DistanceFieldEnum>,
    size : u32,
    bounds: Sphere,
    hash: u64
}
fn rgba_interp(a: Rgba<u8>, b: Rgba<u8>, v: f32) -> Rgba<u8> {
    Rgba([
        ((a[0] as f32) * (1.0 - v) + (b[0] as f32) * v) as u8,
        ((a[1] as f32) * (1.0 - v) + (b[1] as f32) * v) as u8,
        ((a[2] as f32) * (1.0 - v) + (b[2] as f32) * v) as u8,
        ((a[3] as f32) * (1.0 - v) + (b[3] as f32) * v) as u8,
    ])
}
impl Add {
    fn new(left: DistanceFieldEnum, right: DistanceFieldEnum) -> Add {
        Add::from_rc(Rc::new(left), Rc::new(right))
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

    fn color(&self, p: Vec3) -> Rgba<u8> {
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

impl DistanceField for Add {
    fn distance(&self, pos: Vec3) -> f32 {
        f32::min(self.left.distance(pos), self.right.distance(pos))
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Subtract {
     left : Rc<DistanceFieldEnum>,
     subtract: Rc<DistanceFieldEnum>,
     k : f32
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
    pub fn cast_ray(&self, pos: Vec3, dir: Vec3, max_dist: f32) -> Option<(f32, Vec3)> {
        let mut total_distance = 0.0;
        let mut mpos = pos;
        loop {
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
    }

    pub fn optimize_add(add: &Add, block_center: Vec3, size: f32, cache: &mut HashSet<DistanceFieldEnum>, min_d : f32) -> Rc<DistanceFieldEnum> {
        let d2 = add.right.distance(block_center);
        let left_opt = add.left.optimized_for_block2(block_center, size, cache, f32::min(d2, min_d));
        let left_d = left_opt.distance(block_center);
        let right_opt = add.right.optimized_for_block2(block_center, size, cache, f32::min(left_d, min_d));
        
        let right_d = right_opt.distance(block_center);
        if left_d > right_d + size * SQRT3 {
            return right_opt;
        }
        if right_d > left_d + size * SQRT3{
            return left_opt;
        }
        if f32::min(right_d, left_d) > min_d + size * SQRT3 * 2.0 {
            return DistanceFieldEnum::Empty{}.into();
        }
        if f32::min(right_d, left_d) > size * SQRT3 * 2.0 * 1.5 {
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
                if subtract_d > size * SQRT3 * 2.0{
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
                if opt.eq(i) {
                    return Rc::new(self.clone());
                }
                return Rc::new(DistanceFieldEnum::Coloring(c.clone(), opt).into());
            }
            
            _ => return Rc::new(self.clone()),
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
            DistanceFieldEnum::Coloring(c, other) => DistanceFieldEnum::Coloring(c.clone(), Rc::new(other.insert_2(sdf))),
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
                println!("Subtract + add? {}", left2);
                let mut new_left : DistanceFieldEnum = match &left2 {
                    DistanceFieldEnum::Add(inner_add ) => {
                        let left_bounds = inner_add.left.calculate_sphere_bounds();
                        let right_bounds = inner_add.right.calculate_sphere_bounds();
                        if !left_bounds.overlaps(&subbounds) {
                            if !right_bounds.overlaps(&subbounds) {
                                println!("none overlaps! {:?} {:?} {:?} {}", left_bounds, right_bounds, subbounds, (right_bounds.center - subbounds.center).length());
                                // neither bounds overlaps the subtraction -> just delete it.
                                DistanceFieldEnum::Add(inner_add.clone())
                            }else {
                                // left does not overlap -> move the subtraction into the right
                                Add::new(inner_add.left.as_ref().clone(), Subtract::new(inner_add.right.as_ref().clone(), sub.subtract.as_ref().clone(), sub.k).into())
                                 .into()
                            }
                        }else if !right_bounds.overlaps(&subbounds) {
                            // only right overlaps -> move it into the right.
                            Add::new(Subtract::new(inner_add.left.as_ref().clone(), sub.subtract.as_ref().clone(), sub.k).into(),
                               inner_add.right.as_ref().clone()).into()
                        }else{
                            left2
                        }
                        
                    },
                    _=> self.clone()
                };

                if let DistanceFieldEnum::Subtract(sub2) = &new_left{
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

    pub fn color(&self, pos: Vec3) -> Rgba<u8> {
        match self {
            DistanceFieldEnum::Add(add) => add.color(pos),
            DistanceFieldEnum::Primitive(_) => Rgba([255,0,0,255]),
            DistanceFieldEnum::Empty => Rgba([0, 0, 0, 0]),
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
                    return Rc::<DistanceFieldEnum>::ptr_eq(&i,&i2) && b.eq(&b);
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
        
        if !m.contains(self) && m.insert(self.clone()){
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
                DistanceFieldEnum::Empty => todo!(),
            }
        }
    }
    pub fn cached(&self, m : &mut HashSet<DistanceFieldEnum>) -> DistanceFieldEnum{
        
        if !m.contains(self) && m.insert(self.clone()){
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
                DistanceFieldEnum::Empty => todo!(),
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
            _ => {
                core::mem::discriminant(self).hash(state);
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
                    return add2.hash == add.hash// && add.left.eq(&add2.left) && add.right.eq(&add2.right);
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
    let aabb2 = Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0);
    let grad = Noise::new(
        1543,
        Rgba([255, 255, 255, 255]),
        Rgba([100, 140, 150, 255]),
        aabb2.into(),
    );

    let sphere = Sphere::new(Vec3::new(-2.0, 0.0, 0.0), 2.0);

    let noise = Noise::new(
        123,
        Rgba([95, 155, 55, 255]),
        Rgba([255, 255, 255, 255]),
        sphere.into(),
    );

    let sphere2 = Sphere::new(Vec3::new(0.0, -200.0, 0.0), 190.0);
    let noise2 = Noise::new(
        123,
        Rgba([255, 155, 55, 255]),
        Rgba([100, 100, 100, 255]),
        sphere2.into(),
    );

    let sdf = DistanceFieldEnum::Empty {}
        .insert_2(noise)
        .insert_2(grad)
        .insert_2(noise2);
    return sdf;

    
}

pub fn build_test_solid() -> DistanceFieldEnum {
    let aabb2 = Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0).color(Rgba([255, 255, 255, 255]));
    

    let sphere = Sphere::new(Vec3::new(-2.0, 0.0, 0.0), 2.0).color(Rgba([255, 0, 0, 255]));


    let sphere2 = Sphere::new(Vec3::new(0.0, -200.0, 0.0), 190.0).color(Rgba([255, 0, 0, 255]));
    

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
                let sphere = Sphere::new(Vec3::new(*x as f32* 5.0, *y as f32* 5.0, *z as f32 * 5.0), 2.0).color(Rgba([255, 0, 0, 255]));
                sdf = sdf.insert_2(sphere);
            }
        }
    }
    
    return sdf;
}

pub fn build_test3() -> DistanceFieldEnum {
    let mut sdf  = DistanceFieldEnum::Empty;
    sdf = sdf.insert_2(Sphere::new(Vec3::new(0.0,0.0,0.0), 2.0).color(Rgba([255, 0, 0, 255])));
    sdf = sdf.insert_2(Sphere::new(Vec3::new(25.0,0.0,0.0), 2.0).color(Rgba([255, 0, 0, 255])));
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
    sdf = DistanceFieldEnum::Coloring(Coloring::SolidColor(Rgba([0,255,255,255])), Rc::new(sdf));
    return sdf;
}


fn fmt_rgba(rgba: Rgba<u8>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} {} {} {}]", rgba[0], rgba[1], rgba[2], rgba[3])
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
                    Coloring::SolidColor(c) => write!(f, "(color: [{} {} {} {}] {})", c[0], c[1], c[2], c[3], inner),
                    Coloring::Gradient(g) => write!(f, "(gradient: ? {})", inner),
                    Coloring::Noise(_) => write!(f, "(noise: ? {})", inner)
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
    #[test]
    fn test_build_big_optimize2(){
        let sdf = build_big(4);
        let sdfopt = sdf.optimize_bounds();
        
        let bounds =  sdf.calculate_sphere_bounds();
        let mut rng = thread_rng();
        for _ in 0..10000 {
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
            assert!((b - a) / a < e);
        }
    }

    #[test]
    fn test_optimize_subtract(){
        let sdf = DistanceFieldEnum::Empty
            .insert_2(Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0).color(Rgba([0,0,0,1])))
            .insert_2(Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0).color(Rgba([0,0,0,1])));
        let sdf : DistanceFieldEnum = Subtract::new(sdf, Sphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0), 0.0).into();

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
        let sdf = build_big(40);

        let g = sdf.gradient(Vec3::new(-1.0, 0.0, 0.0), 0.1);
        println!("g: {}", g);
    }




}