pub mod mc;
mod sdf;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::rc::Rc;

use kiss3d::resource::{Mesh, Texture, TextureManager};
use sdf::*;
use mc::*;

use kiss3d::light::Light;
use kiss3d::scene::SceneNode;

use kiss3d::window::{State, Window};
use kiss3d::nalgebra::{UnitQuaternion, Vector3, Point3, OPoint, Const, Translation3, Point2, Vector2, DimMul};
use image::{DynamicImage, ImageBuffer, RgbaImage, Rgba};

type Vec3f = Vector3<f32>;
type vec3 = Vec3f;
struct AppState {
    c: SceneNode,
    rot: UnitQuaternion<f32>,
}

impl State for AppState {
    fn step(&mut self, _: &mut Window) {
        self.c.prepend_to_local_rotation(&self.rot)
    }
}

struct VertexesList {
    verts: Vec<Vec3f>
}

impl MarchingCubesReciever for VertexesList{

    fn Receive(&mut self, v1 : Vec3f, v2: Vec3f, v3: Vec3f) {
        self.verts.push(v3);
        self.verts.push(v2);
        self.verts.push(v1);
    }
}
type vec2 = Vector2<f32>;

fn interpolate2(p0 :vec2, p1:vec2, p2:vec2, color0 :vec3, color1:vec3, color2:vec3, point: vec2) -> vec3 {
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

    vec3::new(u * color0.x + v * color1.x + w * color2.x,
        u * color0.y + v * color1.y + w * color2.y,
        u * color0.z + v * color1.z + w * color2.z
    )
}

fn interpolate_color2(v0: &vec2, v1: &vec2, v2: &vec2, 
    c0: &vec3, c1: &vec3, c2: &vec3, point: &vec2) -> vec3 {
let f = |p: &vec2, a: &vec2, b: &vec2| 
((b.y - a.y) * (p.x - a.x) + (a.x - b.x) * (p.y - a.y)) / ((b.y - a.y) * (v0.x - a.x) + (a.x - b.x) * (v0.y - a.y));
let w0 = f(point, v1, v2);
let w1 = f(point, v2, v0);
let w2 = f(point, v0, v1);

w0 * c0 + w1 * c1 + w2 * c2
}

fn interpolate_color(v0: &vec2, v1: &vec2, v2: &vec2,
    c0: &vec3, c1: &vec3, c2: &vec3, point: &vec2) -> vec3 {
let v01 = v1 - v0;
let v02 = v2 - v0;
let v0p = point - v0;

let d = v01.dot(&v02.cross(&v0p));
let u = v0p.dot(&v02.cross(&v0p)) / d;
let v = v01.dot(&v0p.cross(&v0p)) / d;
let w = 1.0 - u - v;

u * c0 + v * c1 + w * c2
}



impl VertexesList{
    fn to_mesh(&self, df : &DistanceFieldEnum) -> (Mesh, DynamicImage){
        let mut coords : Vec<Point3<f32>> = Vec::new();
        let mut faces = Vec::new();
        let mut uvs = Vec::new();
        let mut face : OPoint<u16, Const<3>> = Point3::new(0, 0, 0);
        let mut faceit = 0;
        let mut it = 0;
        let ntriangles :i64 = (self.verts.len() / 3) as i64;
        let columns = f64::ceil(f64::sqrt(f64::try_from(u32::try_from(ntriangles).unwrap()).unwrap())) as i64;
        let rows = ntriangles / columns;
        let fw = 1.0 / (columns as f64);
        let fh = 1.0 / (rows as f64);

        let mut buf = RgbaImage::new(256, 256);
        let bufsize = vec2::new(buf.width() as f32, buf.height() as f32);


        for v in &self.verts {
            let facei: i64 = faces.len() as i64;
            let row = (facei / columns) as f64;
            let col = (facei % columns) as f64;
            let rowf = row / (columns as f64);
            let colf = col / (columns as f64);
            let uv 
                = Point2::new((colf + fw * (match faceit == 1 { false => 0.1, true => 0.9})) as f32 
                     , (rowf + fh * (match faceit == 2 { false => 0.1, true => 0.9})) as f32);
            coords.push(v.clone().into());
            uvs.push(uv);
            face[faceit] = it;
            it += 1;
            faceit += 1;
            if faceit == 3 {
                faceit = 0;
                faces.push(face);

                // now trace the colors into the texture for this triangle.
                let uva : vec2 = uvs[uvs.len() - 3].to_homogeneous().xy();
                let uvb = uvs[uvs.len() - 2].to_homogeneous().xy();
                let uvc = uvs[uvs.len() - 1].to_homogeneous().xy();

                let va = coords[coords.len() - 3].to_homogeneous().xyz();
                let vb = coords[coords.len() - 2].to_homogeneous().xyz();
                let vc = coords[coords.len() - 1].to_homogeneous().xyz();

                let pa = uva * bufsize.x;
                let pb = (uvb * bufsize.x);
                let pc = (uvc * bufsize.x);
                
                for x in (pa.x as u32)+0 ..(pb.x.ceil() as u32 -0) {
                    for y in (pa.y as u32 + 0) .. (pc.y.ceil() as u32 -0) {
                        let p0 = vec2::new(x as f32, y as f32);
                        let v0 = interpolate2(pa, pb, pc, va, vb, vc, p0);
                        //let xx = Vec3f::new(va.x, va.y, va.z) * (1.0 - fx) + Vec3f::new(vb.x, vb.y, vb.z) * fx;
                        let dc = df.distance_color(v0);
                        buf.put_pixel(x,y, dc.1);
                        
                    }
                        
                }
            }
        }
        
        let image = DynamicImage::ImageRgba8(buf);
        

        return (Mesh::new(coords, faces, Option::None, Option::Some(uvs), false), image);
    }
}

static SQRT_3 :f32 = 1.7320508;

fn marching_cubes_sdf<T1: DistanceField, T: MarchingCubesReciever>(recv: &mut T, model : &T1, position : Vec3f, size: f32, res: f32) {
  
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


fn main() {
    let a = vec2::new(0.0,0.0);
    let b = vec2::new(1.0,0.0);
    let c = vec2::new(0.0,1.0);


    let c1 = vec3::new(2.0,0.0, 0.0);
    let c2 = vec3::new(0.0,2.0, 0.0);
    let c3 = vec3::new(0.0,0.0, 2.0);

    let test = vec2::new(0.5, 0.25);
    
    let out = interpolate2(a, b, c, c1, c2, c3, test);
    println!("out: {}", out);
    //return;
    let sdf = DistanceFieldEnum::Empty{}.Insert2(
        Sphere::new(Vec3f::new(-2.0,0.0,0.0), 2.0).color(Rgba([255,0,0,255])))
        
        .Insert2(Sphere::new(Vec3f::new(2.0,0.0,0.0), 2.0).color(Rgba([0, 255,0,255])))
        .Insert2(Sphere::new(Vec3f::new(0.0,2.0,0.0), 2.0).color(Rgba([255, 255,0,255])))
        .Insert2(Sphere::new(Vec3f::new(0.0,-2.0,0.0), 2.0).color(Rgba([255, 255,255,255])))
        .Insert2(Sphere::new(Vec3f::new(0.0,0.0,2.0), 2.0).color(Rgba([255, 0,255,255])))
        .Insert2(Sphere::new(Vec3f::new(0.0,0.0,-2.0), 2.0).color(Rgba([0, 0,255,255])))
        ;

    let d = sdf.distance(Vec3f::new(6.5, 5.0, 0.0));
    println!("distance: {}", d);

    let sdf2 = sdf.optimize_bounds();
    let d2 = sdf2.distance(Vec3f::new(6.5, 5.0, 0.0));

    println!("distance: {}", d2);
    println!("{:?} \n\n {:?}", sdf, sdf2);

    let mut r = VertexesList { verts: Vec::new()};
    //println!("Cube:!\n");
    //process_cube(&sdf, Vec3f::new(1.0, 0.0, 0.0), 0.2, &mut r );

    marching_cubes_sdf(&mut r, &sdf2, Vec3f::zeros(), 5.0, 0.25);
    println!("{:?}", r.verts);



    let mut window = Window::new("Kiss3d: wasm example");

    let mut meshtex = r.to_mesh(&sdf2);
    let mut mesh = meshtex.0;
    let tex = meshtex.1;
    tex.save("test.png");
    let uvs = mesh.uvs();
    
    //println!("UVS: {:?}", uvs.read().unwrap().data().borrow());
    mesh.recompute_normals();
    let mut c = window.add_mesh(Rc::new(RefCell::new(mesh)), Vec3f::new(0.2, 0.2, 0.2));//window.add_cube(0.5, 0.5, 0.5);
    let mut tm = TextureManager::new();
    
    //let mut c = window.add_cube(1.0, 1.0, 1.0);
    //c.add_cube(1.0, 1.0, 1.0).append_translation(&Translation3::new(2.0, 0.0, 0.0));
    c.set_color(1.0, 1.0, 1.0);
    c.append_translation(&Translation3::new(0.0, 0.0, 1.0));
    //c.set_texture_from_memory(&tex, "hello");
    let mut tex2 = tm.add_image(tex, "Hello");
    c.set_texture(tex2);
    

    c.enable_backface_culling(true);
    
    window.set_light(Light::Absolute(Point3::new(0.0,0.0,0.0)));

    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.005);
    let state = AppState { c, rot };

    window.render_loop(state)
}
