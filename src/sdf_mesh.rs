use crate::{
    mc, sdf, triangle_raster,
    vec2::{IntoVector2Array, Vec2},
    vec3::{IntoVector3Array, Vec3},
};

use std::collections::{HashMap, HashSet};

use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use kiss3d::nalgebra::{Const, OPoint, Point3, Vector3};
use kiss3d::resource::Mesh;
use mc::*;
use sdf::*;
use triangle_raster::*;

pub struct TriangleList {
    triangles: Vec<[Vec3; 3]>,
}

impl TriangleList {
    pub fn new() -> TriangleList {
        TriangleList {
            triangles: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    pub fn any(&self) -> bool {
        self.triangles.len() > 0
    }

    pub fn vertices<'a>(&'a self) -> impl Iterator<Item = &'a Vec3> {
        self.triangles.iter().flatten()
    }
}

impl MarchingCubesReciever for TriangleList {
    fn receive(&mut self, v1: Vec3, v2: Vec3, v3: Vec3) {
        self.triangles.push([v1, v2, v3]);
    }
}

pub struct Triangles {
    pub vertices: Vec<Vec3>,
    triangles: Vec<[usize; 3]>, // Indices of vertices forming triangles
}

impl Triangles {
    pub fn from_triangle_list(triangles: &TriangleList) -> Triangles {
        let mut map: HashMap<Vec3, usize> = HashMap::new();
        let mut verts = Vec::new();
        let mut indexes = Vec::new();
        for x in triangles.vertices() {
            let p = x.round(100.0);
            let cur = map.get(&p);
            if cur.is_none() {
                let id = map.len();
                map.insert(p, id);
                verts.push(p);
                indexes.push(id);
            } else {
                indexes.push(*cur.unwrap())
            }
        }

        Triangles {
            vertices: verts,
            triangles: indexes
                .chunks_exact(3)
                .into_iter()
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect(),
        }
    }
    pub fn to_triangle_list(&self) -> TriangleList {
        let mut verts = self
            .triangles
            .iter()
            .map(|x| {
                [
                    self.vertices[x[0]],
                    self.vertices[x[1]],
                    self.vertices[x[2]],
                ]
            })
            .collect();
        TriangleList { triangles: verts }
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn triangle(&mut self, triangle_index: usize) -> Option<[Vec3; 3]> {
        if triangle_index >= self.triangles.len() {
            return None;
        }
        let t = self.triangles[triangle_index];
        return Some([
            self.vertices[t[0]],
            self.vertices[t[1]],
            self.vertices[t[2]],
        ]);
    }

    pub fn triangleidx(&mut self, triangle_index: usize) -> Option<[usize; 3]> {
        if triangle_index >= self.triangles.len() {
            return None;
        }
        let t = self.triangles[triangle_index];
        return Some(t);
    }

    pub fn triangle_size(&self, triangle_index: usize) -> Option<f32> {
        if triangle_index >= self.triangles.len() {
            return None;
        }
        let triangle = self.triangles[triangle_index];
        let a = self.vertices[triangle[0]];
        let b = self.vertices[triangle[1]];
        let c = self.vertices[triangle[2]];

        let la = (a - b).length();
        let lb = (a - c).length();
        let lc = (c - b).length();
        let s = (la + lb + lc) * 0.5;
        let area = (s * (s - la) * (s - lb) * (s - lc)).sqrt();
        return Some(area);
    }

    pub fn collapse_triangle(&mut self, triangle_index: usize) {
        if triangle_index >= self.triangles.len() {
            return; // Or handle the error as needed
        }

        for (i, x) in self.vertices.iter().enumerate() {
            for (i2, x2) in self.vertices.iter().enumerate() {
                if (i == i2) {
                    continue;
                }
                let dother = (*x - *x2).length();
                //assert!(dother >=0.005)
            }
        }

        let triangle = self.triangles[triangle_index];
        let a = self.vertices[triangle[0]];
        let b = self.vertices[triangle[1]];
        let c = self.vertices[triangle[2]];
        let center = (a + b + c) / 3.0;
        self.vertices[triangle[0]] = center.round(100.0);

        self.vertices[triangle[1]] = self.vertices[triangle[0]];
        self.vertices[triangle[2]] = self.vertices[triangle[0]];
        for trig in &mut self.triangles {
            for i in 1..3 {
                if (trig[0] == triangle[i]) {
                    trig[0] = triangle[0];
                }
                if (trig[1] == triangle[i]) {
                    trig[1] = triangle[0];
                }
                if (trig[2] == triangle[i]) {
                    trig[2] = triangle[0];
                }
            }
        }
        // Step 3: Remove the collapsed triangle
        self.triangles.remove(triangle_index);
    }

    pub fn collapse_triangle2(&mut self, triangle_index: usize) {
        // Ensure the triangle index is valid
        if triangle_index >= self.triangles.len() {
            return; // Or handle the error as needed
        }

        let triangle = self.triangles[triangle_index];

        // Check if any vertex is shared by other triangles
        let shared_vertices = self
            .triangles
            .iter()
            .filter(|&t| t != &triangle && t.iter().any(|&v| triangle.contains(&v)));

        // For simplicity, let's collapse the first shared vertex
        if let Some(shared_triangle) = shared_vertices.clone().next() {
            let shared_vertex = shared_triangle.iter().find(|&v| triangle.contains(v));

            if let Some(&shared_vertex_index) = shared_vertex {
                // Update the shared vertices in other triangles to use the remaining vertex
                for t in &mut self.triangles {
                    for v in t.iter_mut() {
                        if *v == shared_vertex_index {
                            *v = triangle
                                .iter()
                                .find(|&t| t != &shared_vertex_index)
                                .cloned()
                                .unwrap();
                        }
                    }
                }

                // Remove the collapsed triangle
                self.triangles.remove(triangle_index);
            }
        }
    }

    pub fn split_triangle(&mut self, triangle_index: usize) {
        // Ensure the triangle index is valid
        if triangle_index >= self.triangles.len() {
            return;
        }

        // Get the indices of the vertices of the triangle
        let triangle = self.triangles[triangle_index];
        let v0 = self.vertices[triangle[0]];
        let v1 = self.vertices[triangle[1]];
        let v2 = self.vertices[triangle[2]];

        // Calculate midpoints of each edge
        let m01 = (v0 + v1) * 0.5;
        let m12 = (v1 + v2) * 0.5;
        let m20 = (v2 + v0) * 0.5;

        // Add midpoints to the vertices list
        let m01_index = self.vertices.len();
        self.vertices.push(m01);
        let m12_index = self.vertices.len();
        self.vertices.push(m12);
        let m20_index = self.vertices.len();
        self.vertices.push(m20);

        // Create new triangles
        self.triangles.push([triangle[0], m01_index, m20_index]);
        self.triangles.push([m01_index, triangle[1], m12_index]);
        self.triangles.push([m20_index, m12_index, triangle[2]]);
        self.triangles.push([m01_index, m12_index, m20_index]);

        // Remove or modify the original triangle
        // This depends on how you want to handle the original triangle.
        // For example, you might want to remove it:
        self.triangles.remove(triangle_index);
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

impl TriangleList {
    pub fn to_mesh(&self, sdf: &DistanceFieldEnum) -> (Mesh, DynamicImage) {
        let mut coords: Vec<Point3<f32>> = Vec::new();
        let mut faces = Vec::new();
        let mut uvs: Vec<Vec2> = Vec::new();
        let mut face: OPoint<u16, Const<3>> = Point3::new(0, 0, 0);
        let mut normals = Vec::new();
        let mut faceit = 0;
        let mut it = 0;
        let ntriangles: i64 = (self.triangles.len()) as i64;
        let columns = f64::ceil(f64::sqrt(
            f64::try_from(u32::try_from(ntriangles).unwrap()).unwrap(),
        )) as i64;
        let rows = ntriangles / columns;
        let fw = 1.0 / (columns as f64);
        let fh = 1.0 / (rows as f64);

        let mut buf: ImageBuffer<Rgba<u8>, Vec<u8>> = RgbaImage::new(512, 512);

        let pxmargin = 3;
        let uvmargin = (1.0 + rows as f64) * (pxmargin as f64) / (buf.width() as f64);
        let bufsize = Vec2::new(buf.width() as f32, buf.height() as f32);
        let mut dict: HashMap<Vector3<i32>, i32> = HashMap::new();
        let mut max = Vec3::new(-100000.0, -100000.0, -100000.0);
        let mut min = Vec3::new(-100000.0, -100000.0, -100000.0);

        for v in self.vertices() {
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

        for v in self.vertices() {
            let facei: i64 = faces.len() as i64;
            let row = (facei / columns) as f64;
            let col = (facei % columns) as f64;
            let rowf = row / (columns as f64);
            let colf = col / (columns as f64);
            let normal = sdf.gradient(*v, 0.03);
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

                let va: Vec3 = coords[coords.len() - 3].to_homogeneous().xyz().into();
                let vb: Vec3 = coords[coords.len() - 2].to_homogeneous().xyz().into();
                let vc: Vec3 = coords[coords.len() - 1].to_homogeneous().xyz().into();

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
                        let dc = sdf2.color(v0).to_u8_rgba();
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
                None,//Option::Some(IntoVector3Array(normals)),
                Option::Some(IntoVector2Array(uvs)),
                false,
            ),
            image,
        );
    }
}

static SQRT_3: f32 = 1.7320508;

pub fn receive_cube(
    recv: &mut TriangleList,
    sdf: &DistanceFieldEnum,
    position: Vec3,
    size: f32, 
    cache : &mut HashSet<DistanceFieldEnum>
) {
    let p = position - Vec3::new(size, size, size);
    let s = size * 2.0;
    let sdf = sdf.optimized_for_block(position, size, cache);

    let mut receive_face = |v1: Vec3, v2: Vec3, v3: Vec3,  v4: Vec3| {
        let a = sdf.distance(v1);
        let b = sdf.distance(v2);
        let c = sdf.distance(v3);
        let d = sdf.distance(v4);
        let limit1 = 0.0;
        let limit2 = s * SQRT_3;
        if a < limit1 || b < limit1 || c < limit1 || d < limit1 {
            return;
        }
        
        //if a > limit2 && b > limit2 && c > limit2 && d > limit2{
        //    return;
        //}
        
        recv.triangles.push([v1, v2, v3]);
        recv.triangles.push([v3, v2, v4]);
    };

    // Front face
    receive_face(p, p + Vec3::new(0.0, s, 0.0), p + Vec3::new(s, 0.0, 0.0), p + Vec3::new(s, s, 0.0));
    {
        // back face.
        let p = p.with_z(p.z + s);
        receive_face(p, p + Vec3::new(s, 0.0, 0.0), p + Vec3::new(0.0, s, 0.0), p + Vec3::new(s, s, 0.0));
    }
    
    // left face
    receive_face(p, p + Vec3::new(0.0, 0.0, s), p + Vec3::new(0.0, s, 0.0), p + Vec3::new(0.0, s, s));
    // right face
    {
        let p = p.with_x(p.x + s);
        receive_face(p, p + Vec3::new(0.0, s, 0.0), p + Vec3::new(0.0, 0.0, s), p + Vec3::new(0.0, s, s));
    
    }

    // bottom face
    receive_face(p, p + Vec3::new(s, 0.0, 0.0), p + Vec3::new(0.0, 0.0 ,s), p + Vec3::new(s, 0.0, s));
    // right face
    {
        let p = p.with_y(p.y + s);
        receive_face(p, p + Vec3::new(0.0, 0.0, s), p + Vec3::new(s, 0.0, 0.0), p + Vec3::new(s, 0.0, s));
    
    }

    
   
}
pub fn marching_voxels(
    recv: &mut TriangleList,
    model: &DistanceFieldEnum,
    position: Vec3,
    size: f32,
    res: f32,
    start_size: f32, 
    cache : &mut HashSet<DistanceFieldEnum>
) {
    let d = model.distance(position);
    // check if the surface collides with a sphere that can include the cube we want to draw.
    if d < size * SQRT_3 {
    
        // If the entire sphere is inside the object, we can skip it.
        
        if d < -size * SQRT_3 {
            return;
        }

        if size <= res {
            receive_cube(recv, model, position, size, cache);
            return;
        }

        let s2 = size * 0.5;
        let o = [-s2, s2];

        for i in 0..8 {
            let offset = Vec3::new(o[i & 1], o[(i >> 1) & 1], o[(i >> 2) & 1]);
            let p = offset + position;
            marching_voxels(recv, model, p, s2, res, start_size, cache);
        }
    }
}
pub fn marching_cubes_sdf<T1: DistanceField>(
    recv: &mut TriangleList,
    model: &T1,
    position: Vec3,
    size: f32,
    res: f32,
    start_size: f32,
) {
    let d = model.distance(position);
    if d < size * SQRT_3  {
        if d < -size * SQRT_3 {
            return;
        }

        if size <= res {
            process_cube(model, position, size, recv);
            return;
        }

        let s2 = size * 0.5;
        let o = [-s2, s2];

        for i in 0..8 {
            let offset = Vec3::new(o[i & 1], o[(i >> 1) & 1], o[(i >> 2) & 1]);
            let p = offset + position;
            marching_cubes_sdf(recv, model, p, s2, res, start_size);
        }
    }
}

fn are_points_on_same_plane<'a, I>(a: Vec3, b: Vec3, c: Vec3, points: I, tolerance: f32) -> bool
where
    I: IntoIterator<Item = &'a Vec3>,
{
    let p0 = a;
    let v1 = b - a;
    let v2 = c - a;
    let normal = v1.cross(v2);

    for point in points {
        let v = Vec3 {
            x: point.x - p0.x,
            y: point.y - p0.y,
            z: point.z - p0.z,
        };

        if v.dot(normal).abs() > tolerance {
            return false;
        }
    }

    true
}

fn cubify_get_faces<T1: DistanceField, T: MarchingCubesReciever>(
    recv: &mut T,
    model: &T1,
    pt: Vec3,
    size: f32,
) {
    let mut cubeindex = 0;

    let isolevel = 0.0;

    let s2 = size * 0.5;
    let o: [f32; 2] = [s2, -s2];

    let mut ds: [f32; 8] = [0.0; 8];
    let mut pts: [Vec3; 8] = Default::default();
    for i in 0..8 {
        let offset = Vec3::new(o[i & 1], o[(i >> 1) & 1], o[(i >> 2) & 1]);
        let offset2 = offset * 2.0;
        let p2 = offset2 + pt;
        let d = model.distance(p2);
        pts[i] = p2;
        ds[i] = d;
    }
    let faces = [
        [0, 1, 2, 3],
        [4, 6, 5, 7],
        [0, 2, 4, 6],
        [1, 5, 3, 7],
        [2, 6, 3, 7],
        [0, 1, 4, 5],
    ];
    // face 0: var x,y fixed z.
    for item in faces.into_iter() {
        if item.iter().all(|x| ds[*x] < -0.1) {
            continue;
        }
        recv.receive(pts[item[0]], pts[item[1]], pts[item[2]]);
        recv.receive(pts[item[1]], pts[item[3]], pts[item[2]]);
    }
}

pub fn cubify<T1: DistanceField, T: MarchingCubesReciever>(
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
            cubify_get_faces(recv, model, position, size);
        } else {
            let s2 = size * 0.5;
            let o = [-s2, s2];
            for i in 0..8 {
                let offset = Vec3::new(o[i & 1], o[(i >> 1) & 1], o[(i >> 2) & 1]);
                let p = offset + position;
                cubify(recv, model, p, s2, res);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_area() {
        let a = Triangles {
            triangles: vec![[0, 1, 2]],
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        };
        let s = a.triangle_size(0).unwrap();
        println!("s: {}", s);
    }
}
