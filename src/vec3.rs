use std::{ops::{Add, Sub, Mul, Div}, fmt};
use kiss3d::nalgebra::{Point3, Vector3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3{
    pub x : f32,
    pub y : f32,
    pub z : f32
}

impl Vec3 {
    // Constructor
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    // Compute the magnitude of the vector
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    // Normalize the vector
    pub fn normalize(&self) -> Self {
        let length = self.length();
        Vec3 {
            x: self.x / length,
            y: self.y / length,
            z: self.z / length,
        }
    }

    pub fn zeros() -> Vec3 { Vec3::new(0.0, 0.0, 0.0) }

    pub fn abs(&self) -> Vec3 { self.map(f32::abs) }

    pub fn length_squared(&self) -> f32 { self.dot(*self) }

    pub fn map<F: FnMut(f32) -> f32>(self, mut f: F) -> Self { Vec3::new(f(self.x), f(self.y), f(self.z)) }
    

    pub fn map2<F: Fn(f32,f32) -> f32>(self, other: Vec3, f: F) -> Self {
        Vec3::new(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))   
    }
    
    pub fn min(self, other: Vec3) -> Vec3{ self.map2(other, f32::min) }

    pub fn max(self, other: Vec3) -> Vec3{ self.map2(other, f32::max) }

    pub fn dot(self, other: Self) -> f32 { self.x * other.x + self.y * other.y + self.z * other.z }
}


impl Add<Vec3> for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Vec3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Div<Vec3> for Vec3 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Vec3 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}


impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, scalar: f32) -> Self {
        Vec3 {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl Add<f32> for Vec3 {
    type Output = Self;

    fn add(self, scalar: f32) -> Self {
        Vec3 {
            x: self.x + scalar,
            y: self.y + scalar,
            z: self.z + scalar,
        }
    }
}

impl Sub<f32> for Vec3 {
    type Output = Self;

    fn sub(self, scalar: f32) -> Self {
        Vec3 {
            x: self.x - scalar,
            y: self.y - scalar,
            z: self.z - scalar,
        }
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3 {
            x: self * vec.x,
            y: self * vec.y,
            z: self * vec.z,
        }
    }
}

impl Mul<&Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, vec: &Vec3) -> Vec3 {
        Vec3 {
            x: self * vec.x,
            y: self * vec.y,
            z: self * vec.z,
        }
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl Into<Vec3> for Vector3<f32>{
    fn into(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

impl Into<Vector3<f32>> for Vec3{
    fn into(self) -> Vector3<f32> {
        Vector3::new(self.x, self.y, self.z)
    }
}
impl Into<Point3<f32>> for Vec3{
    fn into(self) -> Point3<f32> {
        Point3::new(self.x, self.y, self.z)
    }
}
impl Into<Vec3> for Point3<f32>{
    fn into(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

pub fn IntoVector3Array(vecs : Vec<Vec3>) -> Vec<Vector3<f32>> {
    let mut out = Vec::new();
    for x in vecs.into_iter() {
        out.push(x.into());
    }
    return out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let result = a + b;
        assert_eq!(result, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_subtraction() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let result = a - b;
        assert_eq!(result, Vec3::new(-3.0, -3.0, -3.0));
    }

    #[test]
    fn test_scalar_multiplication() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let result = a * 3.0;
        assert_eq!(result, Vec3::new(3.0, 6.0, 9.0));
    }

    #[test]
    fn test_scalar_division() {
        let a = Vec3::new(3.0, 6.0, 9.0);
        let result = a / 3.0;
        assert_eq!(result, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_length() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(a.length(), 5.0);
    }

    #[test]
    fn test_normalize() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        let a = a.normalize();
        let length = (a.x.powi(2) + a.y.powi(2) + a.z.powi(2)).sqrt();
        assert_eq!(length, 1.0);
    }

    #[test]
    fn test_map() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let result = a.map(|x| x * x);
        assert_eq!(result, Vec3::new(1.0, 4.0, 9.0));
    }

    #[test]
    fn test_dot_product() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let dot_product = a.dot(b);
        assert_eq!(dot_product, 32.0);
    }
}
