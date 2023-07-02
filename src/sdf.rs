
use image::Rgba;
use kiss3d::nalgebra::{Vector3};
use std::rc::Rc;

use crate::Vec3;
type Vec3f = Vector3<f32>;

#[derive(Clone, PartialEq, Debug)]
pub enum DistanceFieldEnum{
    Sphere(Sphere),
    Aabb(Aabb),
    Add(Add),
    BoundsAdd(Add, Sphere),
    Gradient(Gradient),
    Empty
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
    color: Rgba<u8>
}

impl Sphere {
    pub fn new(center : Vec3f, radius : f32) -> Sphere{
        Sphere {center: center, radius: radius, color: Rgba([0,0,0,0])}
    }

    pub fn color(&self, color : Rgba<u8> ) -> Sphere {
        Sphere {center : self.center, radius: self.radius, color: color}
    }

    pub fn two_sphere_bounds(a : &Sphere, b: &Sphere) -> Sphere{
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

fn vec3_max(a :Vec3, b : Vec3) -> Vec3 {
    Vec3::new(f32::max(a.x, b.x), f32::max(a.y, b.y), f32::max(a.z, b.z))
}

#[derive(Clone, PartialEq, Debug)]
pub struct Aabb {
    radius : Vec3,
    center : Vec3,
    color : Rgba<u8>
}

impl Aabb {

    pub fn new(center : Vec3, radius: Vec3)-> Aabb{
        Aabb { radius: radius, center: center, color: Rgba([0,0,0,0]) }
    }

    pub fn distance(&self,  p : Vec3f) -> f32{
        let p2 = p - self.center;
        let q = p2.abs() - self.radius;
        return vec3_max(q,Vec3::zeros()).norm()
           + f32::min(f32::max(q.x, f32::max(q.y, q.z)), 0.0);
    }
    
    pub fn color(&self, color : Rgba<u8>) -> Aabb {
        Aabb { radius: self.radius, center: self.center, color: color }
    }
}

impl Into<DistanceFieldEnum> for Aabb{
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Aabb(self)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Gradient {
    p1 : Vec3,
    p2 : Vec3,
    c1 : Rgba<u8>,
    c2 : Rgba<u8>,
    inner : Rc<DistanceFieldEnum>
}

impl Gradient {

    pub fn new(p1 : Vec3, p2 : Vec3, c1: Rgba<u8>, c2: Rgba<u8>, inner: Rc<DistanceFieldEnum>)-> Gradient{
        Gradient {p1 : p1, p2 : p2, c1: c1, c2: c2, inner: inner }
    }
    
    pub fn color(&self, p : Vec3) -> Rgba<u8> {
        let pt2 = p - self.p1;
        let l2 = (self.p1 - self.p2).norm_squared();
        let f = (self.p2 - self.p1).dot(&pt2) / l2;
        return rgba_interp(self.c1, self.c2, f);
          
    }
}

impl Into<DistanceFieldEnum> for Gradient{
    fn into(self) -> DistanceFieldEnum {
        DistanceFieldEnum::Gradient(self)
    }
}


#[derive(Clone, PartialEq, Debug)]
pub struct Add{
    left: Rc<DistanceFieldEnum>,
    right: Rc<DistanceFieldEnum>
}
fn rgba_interp(a : Rgba<u8>, b : Rgba<u8>, v :f32 ) -> Rgba<u8> {
    Rgba([((a[0] as f32) * (1.0 - v) + (b[0] as f32) * v) as u8,
    ((a[1] as f32) * (1.0 - v) + (b[1] as f32) * v) as u8,
    ((a[2] as f32) * (1.0 - v) + (b[2] as f32) * v) as u8,
    ((a[3] as f32) * (1.0 - v) + (b[3] as f32) * v) as u8 ])
}
impl Add {
    fn new(left : DistanceFieldEnum, right : DistanceFieldEnum) -> Add{
        Add { left: Rc::new(left), right: Rc::new(right) }
    }

    fn distance(&self, pos: Vec3f) -> f32 {
        f32::min(self.left.distance(pos), self.right.distance(pos))
    }
    fn distance_color(&self, p : Vec3f) -> (f32, Rgba<u8>){

        let ld = self.left.distance_color(p);
        let rd = self.right.distance_color(p);
        if ld.0 < rd.0 {
            //return (ld.0, rgba_interp(ld.1, rd.1, (ld.0 + 0.1) / (ld.0 + rd.0 + 0.1)));
            
            
            return ld;
        }
        //return (rd.0, rgba_interp(rd.1, ld.1, (rd.0 + 0.1) / (ld.0 + rd.0 + 0.1)));
        
        return rd;
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
impl DistanceField for DistanceFieldEnum{
    fn distance(&self, pos: Vec3f) -> f32 {
        match self {
            DistanceFieldEnum::Add(add) => add.distance(pos),
            DistanceFieldEnum::Sphere(sphere) => sphere.distance(pos),
            DistanceFieldEnum::Aabb(aabb) => aabb.distance(pos),
            DistanceFieldEnum::Empty => f32::INFINITY,
            DistanceFieldEnum::Gradient(gradient) => gradient.inner.distance(pos),
            DistanceFieldEnum::BoundsAdd(add, bounds ) => {
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

impl DistanceFieldEnum{
    pub fn Insert(&self, sdf : DistanceFieldEnum) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Empty => sdf,
            _ => Add::new (self.clone(), sdf).into()

        }
    }
    pub fn Insert2<T : Into<DistanceFieldEnum>>(&self, sdf : T) -> DistanceFieldEnum {
        self.Insert(sdf.into())
    }

    pub fn CalculateSphereBounds(&self) -> Sphere {
        match self {
            DistanceFieldEnum::Sphere(sphere) => sphere.clone(),
            DistanceFieldEnum::Aabb(aabb) => Sphere::new(aabb.center, aabb.radius.norm()),
            DistanceFieldEnum::BoundsAdd(_, sphere ) => sphere.clone(),
            DistanceFieldEnum::Empty => Sphere::new(Vec3f::zeros(), f32::INFINITY),
            DistanceFieldEnum::Gradient(gradient) => gradient.inner.CalculateSphereBounds(),
            DistanceFieldEnum::Add(add) => {
                let left = add.left.CalculateSphereBounds();
                let right = add.right.CalculateSphereBounds();
                
                Sphere::two_sphere_bounds(&left, &right)
            }

        }
    }

    pub fn optimize_bounds(&self) -> DistanceFieldEnum {
        match self {
            DistanceFieldEnum::Add(add) => {
                let left = add.left.optimize_bounds();
                let right = add.right.optimize_bounds();
                let leftbounds = left.CalculateSphereBounds();
                let rightbounds = right.CalculateSphereBounds();
                let bounds = Sphere::two_sphere_bounds(&leftbounds, &rightbounds);
                
                let same = add.left.as_ref().eq(&left) && add.right.as_ref().eq(&right);
                let new_add = match same {true => add.clone(), false => Add::new(left, right)};
                return DistanceFieldEnum::BoundsAdd(new_add, bounds.clone())

            },
            _ => self.clone()
        }
    }

    pub fn distance_color(&self, pos: Vec3f) -> (f32, Rgba<u8>) {
        match self {
            DistanceFieldEnum::Add(add) => add.distance_color(pos),
            DistanceFieldEnum::Sphere(sphere) => (sphere.distance(pos), sphere.color),
            DistanceFieldEnum::Aabb(aabb) => (aabb.distance(pos), aabb.color),
            DistanceFieldEnum::Empty => (f32::INFINITY, Rgba([0,0,0,0])),
            DistanceFieldEnum::BoundsAdd(add, bounds ) => {
                let d1 = bounds.distance(pos);
                if d1 > bounds.radius * 0.5 {
                    return (d1, Rgba([0,0,0,0]))
                }
                add.distance_color(pos)
            },
            DistanceFieldEnum::Gradient(gradient) => (gradient.inner.distance(pos), gradient.color(pos))
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
            let dv =  Vec3::new(dx1 - dx2, dy1 - dy2, dz1 - dz2);
            let l = dv.norm();
            if f32::abs(l) < 0.00001 {
              return Vec3::zeros();
            }
            let x = dv  / l;
            return x;
          }
    

}
