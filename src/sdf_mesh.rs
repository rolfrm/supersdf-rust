use crate::{mc, sdf, triangle_raster, vec3::{Vec3, IntoVector3Array}, vec2::{Vec2, IntoVector2Array}};

use std::collections::{HashMap, HashSet};

use kiss3d::resource::Mesh;
use mc::*;
use sdf::*;
use triangle_raster::*;

use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use kiss3d::nalgebra::{Const, OPoint, Point2, Point3, Vector2, Vector3};

pub struct VertexesList {
    verts: Vec<Vec3>,
}

impl VertexesList {
    pub fn new() -> VertexesList {
        VertexesList { verts: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.verts.len()
    }

    pub fn any(&self) -> bool {
        self.verts.len() > 0
    }
}

impl MarchingCubesReciever for VertexesList {
    fn receive(&mut self, v1: Vec3, v2: Vec3, v3: Vec3) {
        self.verts.push(v3);
        self.verts.push(v2);
        self.verts.push(v1);
    }
}

pub fn interpolate_vec2(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    color0: Vec3,
    color1: Vec3,
    color2: Vec3,
    point: Vec2,
) -> Vec3 {
    let v0 = p1 - p0;
    let v1 = p2 - p0;
    let v2 = point - p0;

    let d00 = v0.x * v0.x + v0.y * v0.y;
    let d01 = v0.x * v1.x + v0.y * v1.y;
    let d11 = v1.x * v1.x + v1.y * v1.y;
    let d20 = v2.x * v0.x + v2.y * v0.y;
    let d21 = v2.x * v1.x + v2.y * v1.y;

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    Vec3::new(
        u * color0.x + v * color1.x + w * color2.x,
        u * color0.y + v * color1.y + w * color2.y,
        u * color0.z + v * color1.z + w * color2.z,
    )
}

fn interpolate_color2(
    v0: &Vec2,
    v1: &Vec2,
    v2: &Vec2,
    c0: &Vec3,
    c1: &Vec3,
    c2: &Vec3,
    point: &Vec2,
) -> Vec3 {
    let f = |p: &Vec2, a: &Vec2, b: &Vec2| {
        ((b.y - a.y) * (p.x - a.x) + (a.x - b.x) * (p.y - a.y))
            / ((b.y - a.y) * (v0.x - a.x) + (a.x - b.x) * (v0.y - a.y))
    };
    let w0 = f(point, v1, v2);
    let w1 = f(point, v2, v0);
    let w2 = f(point, v0, v1);

    w0 * c0 + w1 * c1 + w2 * c2
}

trait VecKey {
    fn key(&self) -> Vector3<i32>;
}

impl VecKey for Vec3 {
    fn key(&self) -> Vector3<i32> {
        Vector3::new(
            (self.x * 100.0) as i32,
            (self.y * 100.0) as i32,
            (self.z * 100.0) as i32,
        )
    }
}

impl VertexesList {
    pub fn to_mesh(&self, sdf: &DistanceFieldEnum) -> (Mesh, DynamicImage) {
        let mut coords: Vec<Point3<f32>> = Vec::new();
        let mut faces = Vec::new();
        let mut uvs : Vec<Vec2> = Vec::new();
        let mut face: OPoint<u16, Const<3>> = Point3::new(0, 0, 0);
        let mut normals = Vec::new();
        let mut faceit = 0;
        let mut it = 0;
        let ntriangles: i64 = (self.verts.len() / 3) as i64;
        let columns = f64::ceil(f64::sqrt(
            f64::try_from(u32::try_from(ntriangles).unwrap()).unwrap(),
        )) as i64;
        let rows = ntriangles / columns;
        let fw = 1.0 / (columns as f64);
        let fh = 1.0 / (rows as f64);

        let mut buf: ImageBuffer<Rgba<u8>, Vec<u8>> = RgbaImage::new(128, 128);
        let pxmargin = 3;
        let uvmargin = (1.0 + rows as f64) * (pxmargin as f64) / (buf.width() as f64);
        let bufsize = Vec2::new(buf.width() as f32, buf.height() as f32);
        let mut dict: HashMap<Vector3<i32>, i32> = HashMap::new();
        let mut max = Vec3::new(-100000.0, -100000.0, -100000.0);
        let mut min = Vec3::new(-100000.0, -100000.0, -100000.0);

        for v in &self.verts {
            min = min.min(*v);
            max = max.max(*v);
            let key = v.key();

            if dict.contains_key(&key) {
                dict.insert(key, dict.len() as i32);
            }
        }
        let mid = (max + min) * 0.5;
        let size = (max - min).length() * 0.5;

        let mut cache = HashSet::new();

        let sdf = sdf.optimized_for_block(mid, size, &mut cache);


        for v in &self.verts {
            let facei: i64 = faces.len() as i64;
            let row = (facei / columns) as f64;
            let col = (facei % columns) as f64;
            let rowf = row / (columns as f64);
            let colf = col / (columns as f64);
            let normal = sdf.gradient(*v, 0.01);
            normals.push(normal);

            let uv = Vec2::new(
                (colf
                    + fw * (match faceit == 1 {
                        false => uvmargin,
                        true => 1.0 - uvmargin,
                    })) as f32,
                (rowf
                    + fh * (match faceit == 2 {
                        false => uvmargin,
                        true => 1.0 - uvmargin,
                    })) as f32,
            );
            coords.push(v.clone().into());
            uvs.push(uv);
            face[faceit] = it;

            it += 1;
            faceit += 1;
            if faceit == 3 {
                faceit = 0;
                faces.push(face);

                let va : Vec3 = coords[coords.len() - 3].to_homogeneous().xyz().into();
                let vb : Vec3 = coords[coords.len() - 2].to_homogeneous().xyz().into();
                let vc : Vec3 = coords[coords.len() - 1].to_homogeneous().xyz().into();

                // now trace the colors into the texture for this triangle.
                let uva: Vec2 = uvs[uvs.len() - 3];
                let uvb = uvs[uvs.len() - 2];
                let uvc = uvs[uvs.len() - 1];

                // uv in pixel coordinate with a 1px margin.
                let pa = uva * bufsize.x;
                let pb = uvb * bufsize.x;
                let pc = uvc * bufsize.x;
                let trig = Triangle::new(
                    pa + Vec2::new(-2.0, -2.0),
                    pb + Vec2::new(2.0, -2.0),
                    pc + Vec2::new(-2.0, 2.0),
                );
                let min2 = va.min(vb).min(vc);
                let max2 = va.max(vb).max(vc);
                let mid2 = (max2 + min2) * 0.5;
                let size2 = (max2 - min2).length() * 0.5;
                let sdf2 = sdf.optimized_for_block(mid2, size2, &mut cache);
                
                iter_triangle(&trig, |pixel| {
                    let x = pixel.x as u32;
                    let y = pixel.y as u32;
                    let px2 = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let v0 = interpolate_vec2(pa, pb, pc, va, vb, vc, px2);

                    if x < buf.width() && y < buf.height() {
                        let dc = sdf2.color(v0);
                        buf.put_pixel(x, y, dc);
                    }
                });
            }
        }

        let image = DynamicImage::ImageRgba8(buf);

        return (
            Mesh::new(
                coords,
                faces,
                Option::Some(IntoVector3Array(normals)),
                Option::Some(IntoVector2Array(uvs)),
                false,
            ),
            image,
        );
    }
}

static SQRT_3: f32 = 1.7320508;

pub fn marching_cubes_sdf<T1: DistanceField, T: MarchingCubesReciever>(
    recv: &mut T,
    model: &T1,
    position: Vec3,
    size: f32,
    res: f32,
) {
    let d = model.distance(position);
    if d < size * SQRT_3 {
        if d < -size * SQRT_3 {
            return;
        }
        if size <= res {
            process_cube(model, position, size, recv);
        } else {
            let s2 = size * 0.5;
            let o = [-s2, s2];
            for i in 0..8 {
                let offset = Vec3::new(o[i & 1], o[(i >> 1) & 1], o[(i >> 2) & 1]);
                let p = offset + position;
                marching_cubes_sdf(recv, model, p, s2, res);
            }
        }
    }
}
