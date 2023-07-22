use image::{Pixel, Rgba};
use kiss3d::nalgebra as na;
use kiss3d::nalgebra::{Vector3};
use noise::{NoiseFn, Perlin};//, Simplex, SuperSimplex};
use std::rc::Rc;
type Vec3 = Vector3<f32>;
type Vec3f = Vec3;

const sqrt_3: f32 = 1.73205080757;

#[derive(Clone, PartialEq, Debug)]
pub enum DistanceFieldEnum {
    Sphere(Sphere),
    Aabb(Aabb),
    Add(Add),
    BoundsAdd(Add, Sphere),
    Gradient(Gradient),
    Noise(Noise),
    Subtract(Subtract),
    Empty,
}

pub trait DistanceField {
    fn distance(&self, pos: Vec3f) -> f32 {
        return f32::INFINITY;
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Sphere {
    center: Vec3f,
    radius: f32,
    color: Rgba<u8>,
}

impl Sphere {
    pub fn new(center: Vec3f, radius: f32) -> Sphere {
        Sphere {
            center: center,
            radius: radius,
            color: Rgba([0, 0, 0, 0]),
        }
    }

    pub fn color(&self, color: Rgba<u8>) -> Sphere {
        Sphere {
            center: self.center,
            radius: self.radius,
            color: color,
        }
    }

    pub fn two_sphere_bounds(a: &Sphere, b: &Sphere) -> Sphere {
        let r2ld = (a.center - b.center).normalize();
        let leftext = a.center + r2ld * a.radius;
        let rightext = b.center - r2ld * b.radius;

        let center = (leftext + rightext) * 0.5;
        let radius = (leftext - rightext).norm() * 0.5;

        return Sphere::new(center, radius);
    }
}

impl Into<DistanceFieldEnum> for Sphere {
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Sphere(self)
    }
}

fn vec3_max(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(f32::max(a.x, b.x), f32::max(a.y, b.y), f32::max(a.z, b.z))
}

#[derive(Clone, PartialEq, Debug)]
pub struct Aabb {
    radius: Vec3,
    center: Vec3,
    color: Rgba<u8>
}

impl Aabb {
    pub fn new(center: Vec3, radius: Vec3) -> Aabb {
        Aabb {
            radius: radius,
            center: center,
            color: Rgba([0, 0, 0, 0]),
        }
    }

    pub fn distance(&self, p: Vec3f) -> f32 {
        let p2 = p - self.center;
        let q = p2.abs() - self.radius;
        return vec3_max(q, Vec3::zeros()).norm()
            + f32::min(f32::max(q.x, f32::max(q.y, q.z)), 0.0);
    }

    pub fn color(&self, color: Rgba<u8>) -> Aabb {
        Aabb {
            radius: self.radius,
            center: self.center,
            color: color,
        }
    }
}

impl Into<DistanceFieldEnum> for Aabb {
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Aabb(self)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Gradient {
    p1: Vec3,
    p2: Vec3,
    c1: Rgba<u8>,
    c2: Rgba<u8>,
    inner: Rc<DistanceFieldEnum>,
}

impl Gradient {
    pub fn new(
        p1: Vec3,
        p2: Vec3,
        c1: Rgba<u8>,
        c2: Rgba<u8>,
        inner: Rc<DistanceFieldEnum>,
    ) -> Gradient {
        Gradient {
            p1: p1,
            p2: p2,
            c1: c1,
            c2: c2,
            inner: inner,
        }
    }

    pub fn color(&self, p: Vec3) -> Rgba<u8> {
        let pt2 = p - self.p1;
        let l2 = (self.p1 - self.p2).norm_squared();
        let f = (self.p2 - self.p1).dot(&pt2) / l2;
        let mut colorbase = self.inner.color(p);
        let mut color = rgba_interp(self.c1, self.c2, f);
        colorbase.blend(&color);
        colorbase.blend(&color);
        return colorbase;
    }
}

impl Into<DistanceFieldEnum> for Gradient {
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Gradient(self)
    }
}

#[derive(Clone, Debug)]
pub struct Noise {
    noise: Rc<Perlin>,
    seed: u32,
    c1: Rgba<u8>,
    c2: Rgba<u8>,
    inner: Rc<DistanceFieldEnum>,
}
impl PartialEq for Noise {
    fn eq(&self, other: &Self) -> bool {
        return self.seed == other.seed
            && self.c1 == other.c1
            && self.c2 == other.c2
            && self.inner == other.inner;
    }
}

impl Noise {
    pub fn new(seed: u32, c1: Rgba<u8>, c2: Rgba<u8>, inner: DistanceFieldEnum) -> Noise {
        Noise {
            seed: seed,
            noise: Rc::new(Perlin::new(seed)),
            c1: c1,
            c2: c2,
            inner: Rc::new(inner),
        }
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
        let mut color = rgba_interp(self.c1, self.c2, 0.5 * (n1 + n2 + n3) as f32);
        if color[3] < 255 {
            let mut colorbase = self.inner.color(pos);
            colorbase.blend(&color);
            return colorbase;
        }
        return color;
    }
}

impl Into<DistanceFieldEnum> for Noise {
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Noise(self)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Add {
    left: Rc<DistanceFieldEnum>,
    right: Rc<DistanceFieldEnum>
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
        Add {
            left: Rc::new(left),
            right: Rc::new(right),
        }
    }

    fn distance(&self, pos: Vec3f) -> f32 {
        f32::min(self.left.distance(pos), self.right.distance(pos))
    }

    fn color(&self, p: Vec3f) -> Rgba<u8> {
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
    fn distance(&self, pos: Vec3f) -> f32 {
        (pos - self.center).norm() - self.radius
    }
}
impl DistanceField for DistanceFieldEnum {
    fn distance(&self, pos: Vec3f) -> f32 {
        match self {
            DistanceFieldEnum::Add(add) => add.distance(pos),
            DistanceFieldEnum::Subtract(sub) => sub.distance(pos),
            DistanceFieldEnum::Sphere(sphere) => sphere.distance(pos),
            DistanceFieldEnum::Aabb(aabb) => aabb.distance(pos),
            DistanceFieldEnum::Empty => f32::INFINITY,
            DistanceFieldEnum::Gradient(gradient) => gradient.inner.distance(pos),
            DistanceFieldEnum::Noise(noise) => noise.inner.distance(pos),
            DistanceFieldEnum::BoundsAdd(add, bounds) => {
                let d1 = bounds.distance(pos);
                if d1 > bounds.radius * 0.5 {
                    return d1;
                }
                return add.distance(pos);
            }
        }
    }
}

impl DistanceField for Add {
    fn distance(&self, pos: Vec3f) -> f32 {
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
    fn distance(&self, pos : Vec3f) -> f32 {
    let k = self.k;
    let d1 = self.left.distance(pos);
    let d2 = self.subtract.distance(pos);
    let h = f32::clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    
    let r = f32mixf(d1, -d2, h) + k * h * (1.0 - h);
    return r;
    }
}

impl Subtract {
    pub fn new<T : Into<DistanceFieldEnum>, T2 : Into<DistanceFieldEnum>>(left : T, right : T2, k : f32) -> Subtract {
        Subtract { left: Rc::new(left.into()), subtract: Rc::new(right.into()), k: k }
    }
}
impl Into<DistanceFieldEnum> for Subtract {
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Subtract(self)
    }
}
impl DistanceFieldEnum {
    pub fn cast_ray(&self, pos: Vec3f, dir: Vec3f, max_dist: f32) -> Option<(f32, Vec3f)> {
        let mut total_distance = 0.0;
        let mut mpos = pos;
        println!("{:?} {:?}", pos, dir);
        loop {
            let d = self.distance(mpos);
            println!("{:?} {:?}", pos, d);

            total_distance += d;
            mpos = mpos + d * dir;
            if total_distance > max_dist {
                return None;
            }
            if d < 0.001 {
                return Some((total_distance, mpos));
            }
        }
    }

    pub fn optimize_add(add: &Add, block_center: Vec3f, size: f32) -> DistanceFieldEnum {
        let left_opt = add.left.optimized_for_block(block_center, size);
        let right_opt = add.right.optimized_for_block(block_center, size);
        let left_d = left_opt.distance(block_center);
        let right_d = right_opt.distance(block_center);
        if left_d > right_d + size * sqrt_3 {
            return right_opt;
        }
        if right_d > left_d + size * sqrt_3 {
            return left_opt;
        }
        if left_opt.eq(&add.left) && right_opt.eq(&add.right) {
            let add2 = add.clone();
            //println!("Reuse! {}", Rc::strong_count(&add2.left));
            return DistanceFieldEnum::Add(add2);
        }
        return Add::new(left_opt, right_opt).into();
    }

    pub fn optimized_for_block(&self, block_center: Vec3f, size: f32) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Add(add) => {
                DistanceFieldEnum::optimize_add(add, block_center,size)
            },
            DistanceFieldEnum::Subtract(sub) => {
                let subtract_d = sub.subtract.distance(block_center);
                let left2 = sub.left.optimized_for_block(block_center, size);
                if subtract_d > size * sqrt_3 * 1.5 {
                    if left2.eq(&sub.left) {
                        return sub.left.as_ref().clone();
                    }
                    return left2;
                }
                if left2.eq(&sub.left) {
                    return DistanceFieldEnum::Subtract(sub.clone());
                }
                DistanceFieldEnum::Subtract( Subtract {left : Rc::new(left2), subtract: sub.subtract.clone(), k: sub.k}).into()
                
            }

            DistanceFieldEnum::BoundsAdd(add, bounds) => {
                let bounds_d = bounds.distance(block_center);
                if bounds_d < size * sqrt_3 {
                    let r = DistanceFieldEnum::optimize_add(add, block_center, size);
                    match r {
                        DistanceFieldEnum::Add(add) =>
                            DistanceFieldEnum::BoundsAdd(add, bounds.clone()),
                        _ =>
                        r

                    }
                    
                }else{
                    self.clone()
                }

            }
            
            _ => return self.clone(),
        }
    }

    pub fn distance_and_optiomize(&self, pos: Vec3f, size: f32) -> (f32, DistanceFieldEnum) {
        let pos2 = na::Matrix::map(&pos, |x| f32::floor(x / size) * size);
        let sdf2 =
            self.optimized_for_block(pos2 + Vec3::new(size * 0.5, size * 0.5, size * 0.5), size);
        
        return (self.distance(pos), sdf2);
    }

    pub fn Insert(&self, sdf: DistanceFieldEnum) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Empty => sdf,
            _ => Add::new(self.clone(), sdf).into(),
        }
    }
    pub fn Insert2<T: Into<DistanceFieldEnum>>(&self, sdf: T) -> DistanceFieldEnum {
        self.Insert(sdf.into())
    }

    pub fn calculate_sphere_bounds(&self) -> Sphere {
        match self {
            DistanceFieldEnum::Sphere(sphere) => sphere.clone(),
            DistanceFieldEnum::Aabb(aabb) => Sphere::new(aabb.center, aabb.radius.norm()),
            DistanceFieldEnum::BoundsAdd(_, sphere) => sphere.clone(),
            DistanceFieldEnum::Empty => Sphere::new(Vec3f::zeros(), f32::INFINITY),
            DistanceFieldEnum::Gradient(gradient) => gradient.inner.calculate_sphere_bounds(),
            DistanceFieldEnum::Noise(noise) => noise.inner.calculate_sphere_bounds(),
            DistanceFieldEnum::Add(add) => {
                let left = add.left.calculate_sphere_bounds();
                let right = add.right.calculate_sphere_bounds();

                Sphere::two_sphere_bounds(&left, &right)
            },
            DistanceFieldEnum::Subtract(sub) => sub.left.calculate_sphere_bounds()
        }
    }

    pub fn optimize_bounds(&self) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Add(add) => {
                let left = add.left.optimize_bounds();
                let right = add.right.optimize_bounds();
                let leftbounds = left.calculate_sphere_bounds();
                let rightbounds = right.calculate_sphere_bounds();
                let bounds = Sphere::two_sphere_bounds(&leftbounds, &rightbounds);

                let same = add.left.as_ref().eq(&left) && add.right.as_ref().eq(&right);
                let new_add = match same {
                    true => add.clone(),
                    false => Add::new(left, right),
                };
                return DistanceFieldEnum::BoundsAdd(new_add, bounds.clone());
            }
            _ => self.clone(),
        }
    }

    pub fn color(&self, pos: Vec3f) -> Rgba<u8> {
        match self {
            DistanceFieldEnum::Add(add) => add.color(pos),
            DistanceFieldEnum::Sphere(sphere) => sphere.color,
            DistanceFieldEnum::Aabb(aabb) => aabb.color,
            DistanceFieldEnum::Empty => Rgba([0, 0, 0, 0]),
            DistanceFieldEnum::BoundsAdd(add, bounds) => {
                let d1 = bounds.distance(pos);
                if d1 > bounds.radius * 0.5 {
                    return Rgba([0, 0, 0, 0]);
                }
                add.color(pos)
            }
            DistanceFieldEnum::Gradient(gradient) => gradient.color(pos),
            
            DistanceFieldEnum::Noise(noise) => noise.color(pos),
            DistanceFieldEnum::Subtract(sub) => sub.left.color(pos)
        }
    }

    pub fn gradient(&self, pos: Vec3f, size: f32) -> Vec3f {
        let mut ptx = pos;
        let mut pty = pos;
        let mut ptz = pos;
        ptx.x += size * 0.2;
        pty.y += size * 0.2;
        ptz.z += size * 0.2;
        let dx1 = self.distance(ptx);
        let dy1 = self.distance(pty);
        let dz1 = self.distance(ptz);

        ptx.x -= size * 0.2 * 2.0;
        pty.y -= size * 0.2 * 2.0;
        ptz.z -= size * 0.2 * 2.0;
        let dx2 = self.distance(ptx);
        let dy2 = self.distance(pty);
        let dz2 = self.distance(ptz);
        let dv = Vec3::new(dx1 - dx2, dy1 - dy2, dz1 - dz2);
        let l = dv.norm();
        if f32::abs(l) < 0.00001 {
            return Vec3::zeros();
        }
        let x = dv / l;
        return x;
    }
}


pub fn build_test() -> DistanceFieldEnum {
    let aabb2 = Sphere::new(Vec3f::new(2.0, 0.0, 0.0), 1.0).color(Rgba([255, 255, 255, 255]));
    let grad = Noise::new(
        1543,
        Rgba([255, 255, 255, 255]),
        Rgba([100, 140, 150, 255]),
        aabb2.into(),
    );

    let sphere = Sphere::new(Vec3f::new(-2.0, 0.0, 0.0), 2.0).color(Rgba([255, 0, 0, 255]));

    let noise = Noise::new(
        123,
        Rgba([95, 155, 55, 255]),
        Rgba([255, 255, 255, 255]),
        sphere.into(),
    );

    let sphere2 = Sphere::new(Vec3f::new(0.0, -200.0, 0.0), 190.0).color(Rgba([255, 0, 0, 255]));
    let noise2 = Noise::new(
        123,
        Rgba([255, 155, 55, 255]),
        Rgba([100, 100, 100, 255]),
        sphere2.into(),
    );

    let sdf = DistanceFieldEnum::Empty {}
        .Insert2(noise)
        .Insert2(grad)
        .Insert2(noise2);
    return sdf;

    
}


pub fn build_test2() -> DistanceFieldEnum {
    let mut sdf  = DistanceFieldEnum::Empty;
    for x in 0..10 {
        for y in 0..10 {
            for z in 0..10 {
                let sphere = Sphere::new(Vec3f::new(x as f32* 5.0, y as f32* 5.0, z as f32 * 5.0), 2.0).color(Rgba([255, 0, 0, 255]));
                sdf = sdf.Insert2(sphere);
            }
        }
    }
    
    return sdf;

    
}


#[cfg(test)]
mod tests {
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


}