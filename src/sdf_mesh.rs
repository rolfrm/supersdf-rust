use crate::{sdf, mc, triangle_raster};

use std::collections::HashMap;

use kiss3d::{resource::{Texture, TextureManager, Mesh}, nalgebra::ComplexField};
use sdf::*;
use mc::*;
use triangle_raster::*;

use kiss3d::nalgebra::{Vector3, Point3, OPoint, Const, Point2, Vector2};
use image::{DynamicImage, ImageBuffer, RgbaImage, Rgba};

type Vec3f = Vector3<f32>;
type Vec3 = Vec3f;
type Vec2 = Vector2<f32>;


pub struct VertexesList {
    verts: Vec<Vec3f>
}

impl VertexesList {
    pub fn new() -> VertexesList{
        VertexesList { verts: Vec::new() }
    }

    pub fn any(&self) -> bool {
        self.verts.len() > 0
    }
}

impl MarchingCubesReciever for VertexesList{

    fn Receive(&mut self, v1 : Vec3f, v2: Vec3f, v3: Vec3f) {
        self.verts.push(v3);
        self.verts.push(v2);
        self.verts.push(v1);
    }
}

pub fn interpolate_vec2(p0 :Vec2, p1:Vec2, p2:Vec2, color0 :Vec3, color1:Vec3, color2:Vec3, point: Vec2) -> Vec3 {
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

    Vec3::new(u * color0.x + v * color1.x + w * color2.x,
        u * color0.y + v * color1.y + w * color2.y,
        u * color0.z + v * color1.z + w * color2.z
    )
}

fn interpolate_color2(v0: &Vec2, v1: &Vec2, v2: &Vec2, 
    c0: &Vec3, c1: &Vec3, c2: &Vec3, point: &Vec2) -> Vec3 {
let f = |p: &Vec2, a: &Vec2, b: &Vec2| 
((b.y - a.y) * (p.x - a.x) + (a.x - b.x) * (p.y - a.y)) / ((b.y - a.y) * (v0.x - a.x) + (a.x - b.x) * (v0.y - a.y));
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
        Vector3::new((self.x * 100.0) as i32, (self.y * 100.0) as i32, (self.z * 100.0) as i32)
    }
}



impl VertexesList{
    pub fn to_mesh(&self, df : &DistanceFieldEnum) -> (Mesh, DynamicImage){
        let mut coords : Vec<Point3<f32>> = Vec::new();
        let mut faces = Vec::new();
        let mut uvs = Vec::new();
        let mut face : OPoint<u16, Const<3>> = Point3::new(0, 0, 0);
        let mut normals = Vec::new();
        let mut faceit = 0;
        let mut it = 0;
        let ntriangles :i64 = (self.verts.len() / 3) as i64;
        let columns = f64::ceil(f64::sqrt(f64::try_from(u32::try_from(ntriangles).unwrap()).unwrap())) as i64;
        let rows = ntriangles / columns;
        let fw = 1.0 / (columns as f64);
        let fh = 1.0 / (rows as f64);

        let mut buf: ImageBuffer<Rgba<u8>, Vec<u8>> = RgbaImage::new(1024, 1024);
        let pxmargin = 3;
        let uvmargin = (1.0 + rows as f64) * (pxmargin as f64) / (buf.width() as f64);
        let bufsize: kiss3d::nalgebra::Matrix<f32, Const<2>, Const<1>, kiss3d::nalgebra::ArrayStorage<f32, 2, 1>> = Vec2::new(buf.width() as f32, buf.height() as f32);
        let mut dict : HashMap<Vector3<i32>, i32> = HashMap::new();

        let mut edges : HashMap<Vector2<i32>, i32> = HashMap::new();

        for v in &self.verts {
            let key = v.key();
            
            if dict.contains_key(&key) {
                dict.insert(key, dict.len() as i32);
            }
        }


        let margin = 0.1;
        for v in &self.verts {
            let facei: i64 = faces.len() as i64;
            let row = (facei / columns) as f64;
            let col = (facei % columns) as f64;
            let rowf = row / (columns as f64);
            let colf = col / (columns as f64);
            let normal = df.gradient(*v, 0.001);
            normals.push(normal);
            
            let uv 
                = Point2::new((colf +  fw * (match faceit == 1 { false => uvmargin, true => 1.0 - uvmargin})) as f32 
                     , (rowf + fh * (match faceit == 2 { false => uvmargin, true => 1.0 - uvmargin})) as f32);
            coords.push(v.clone().into());
            uvs.push(uv);
            face[faceit] = it;
            it += 1;
            faceit += 1;
            if faceit == 3 {
                faceit = 0;
                faces.push(face);

                let va = coords[coords.len() - 3].to_homogeneous().xyz();
                let vb = coords[coords.len() - 2].to_homogeneous().xyz();
                let vc = coords[coords.len() - 1].to_homogeneous().xyz();
                
                let ka = va.key();
                let kb = vb.key();
                let kc = vc.key();
                /*let ia = dict[&ka];
                let ib = dict[&kb];
                let ic = dict[&kc];

                let e1 = if ia < ib {Vector2::new(ia, ib)} else {Vector2::new(ib, ia)};
                let e2 = if ia < ic {Vector2::new(ia, ic)} else {Vector2::new(ic, ia)};
                let e3 = if ic < ib {Vector2::new(ic, ib)} else {Vector2::new(ib, ic)};
                if edges.contains_key(&e1) {
                    edges.insert(e1, edges.len() as i32);
                }                
                if edges.contains_key(&e2) {
                    edges.insert(e2, edges.len() as i32);
                }                
                if edges.contains_key(&e3) {
                    edges.insert(e3, edges.len() as i32);
                } */               

                // now trace the colors into the texture for this triangle.
                let uva : Vec2 = uvs[uvs.len() - 3].to_homogeneous().xy();
                let uvb = uvs[uvs.len() - 2].to_homogeneous().xy();
                let uvc = uvs[uvs.len() - 1].to_homogeneous().xy();


                // uv in pixel coordinate with a 1px margin.
                let pa = uva * bufsize.x;
                let pb = uvb * bufsize.x;
                let pc = uvc * bufsize.x;
                let trig = Triangle::new(
                    pa + Vec2::new(-2.0, -2.0), 
                    pb + Vec2::new(2.0, -2.0), 
                    pc + Vec2::new(-2.0, 2.0)
                );
                println!("{:?} {}", trig, uvmargin);
                    

                iter_triangle(&trig, |pixel| {
                    let x = pixel.x as u32;
                    let y = pixel.y as u32;
                    let px2 = Vec2::new(x as f32, y as f32) + Vec2::new(0.5, 0.5);
                    let v0 = interpolate_vec2(pa, pb, pc, va, vb, vc, px2);
                        
                    let dc = df.distance_color(v0);
                    if x < buf.width() && y < buf.height() {
                        buf.put_pixel(x,y, dc.1);
                    }

                });
            }
        }
        
        let image = DynamicImage::ImageRgba8(buf);
        

        return (Mesh::new(coords, faces, Option::Some(normals), 
        Option::Some(uvs), false), image);
    }
}

static SQRT_3 :f32 = 1.7320508;

pub fn marching_cubes_sdf<T1: DistanceField, T: MarchingCubesReciever>(recv: &mut T, model : &T1, position : Vec3f, size: f32, res: f32) {
  
  let d = model.distance(position);
  if (d < size * SQRT_3) {
	 if(d < -size * SQRT_3){
		return;
     }
    if (size <= res) {
      process_cube(model, position, size, recv);
    } else {

      let s2 = size * 0.5;
      let o = [-s2, s2];
      for i in 0..8 {
        let offset = Vec3f::new(o[i & 1], o[(i >> 1) & 1], o[(i >> 2) & 1]);
        let p = offset + position;
        marching_cubes_sdf(recv, model, p, s2, res);
      }
    }
  }
}

