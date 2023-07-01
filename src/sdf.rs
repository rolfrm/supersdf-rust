
use image::Rgba;
use kiss3d::nalgebra::{Vector3};
use std::rc::Rc;
type Vec3f = Vector3<f32>;

#[derive(Clone, PartialEq, Debug)]
pub enum DistanceFieldEnum{
    Sphere(Sphere),
    Add(Add),
    BoundsAdd(Add, Sphere),
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
            if(ld.0 > 0.0){
               //return (ld.0, rgba_interp(ld.1, rd.1, ld.0 / (ld.0 + rd.0)));
            }
            
            return ld;
        }
        if(rd.0 > 0.0){
            //return (rd.0, rgba_interp(rd.1, ld.1, rd.0 / (ld.0 + rd.0)));
        }
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
            DistanceFieldEnum::Empty => f32::INFINITY,
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
            DistanceFieldEnum::BoundsAdd(_, sphere ) => sphere.clone(),
            DistanceFieldEnum::Empty => Sphere::new(Vec3f::zeros(), f32::INFINITY),
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
            DistanceFieldEnum::Empty => (f32::INFINITY, Rgba([0,0,0,0])),
            DistanceFieldEnum::BoundsAdd(add, bounds ) => {
                let d1 = bounds.distance(pos);
                if d1 > bounds.radius * 0.5 {
                    return (d1, Rgba([0,0,0,0]))
                }
                add.distance_color(pos)
            }
        }
    }

}
