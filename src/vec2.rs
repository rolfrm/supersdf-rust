use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn abs(&self) -> Vec2 {
        Vec2 { x: self.x.abs(), y: self.y.abs() }
    }

    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl Add<Vec2> for Vec2 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }
}

impl Sub<Vec2> for Vec2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Vec2 { x: self.x - other.x, y: self.y - other.y }
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Vec2 { x: self.x * s, y: self.y * s }
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    fn div(self, s: f32) -> Self {
        Vec2 { x: self.x / s, y: self.y / s }
    }
}

impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Vec2 { x: -self.x, y: -self.y }
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
