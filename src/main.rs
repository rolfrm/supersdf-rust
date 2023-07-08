mod mc;
mod sdf;
mod sdf_mesh;
mod triangle_raster;

use noise::{Perlin, NoiseFn};

use std::cell::RefCell;
use std::rc::Rc;

use kiss3d::resource::TextureManager;
use sdf::*;
use sdf_mesh::*;

use kiss3d::light::Light;
use kiss3d::scene::SceneNode;

use kiss3d::window::{State, Window};
use kiss3d::nalgebra::{UnitQuaternion, Vector3, Translation3, Vector2};
use image::{Rgba, ImageBuffer, RgbaImage, DynamicImage};

use crate::triangle_raster::Triangle;

type Vec3f = Vector3<f32>;
type Vec3 = Vec3f;
type Vec2 = Vector2<f32>;

struct AppState {
    c: SceneNode,
    rot: UnitQuaternion<f32>,
}

impl State for AppState {
    fn step(&mut self, _: &mut Window) {
        self.c.prepend_to_local_rotation(&self.rot)
    }
}

fn main() {

    let n1 = Perlin::default();
    for i in -50..50 {
        let v0 = (i as f64) * 0.2;
        let n = n1.get([v0, 0.5, 9.0]);
        let n2 = n1.get([v0, 0.5, 9.0]);
        println!("{}: {} {}", v0, n, n2);
    }
    
    let i = sdf_mesh::interpolate_vec2(Vec2::new(0.0,0.0), Vec2::new(1.0,0.0), Vec2::new(0.0,1.0), 
        Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 
        Vec2::new(-2.5, 0.5));
    
    println!("{:?}", i);
    //return;
    let line = triangle_raster::Line::new(Vec2::new(1.0, 0.5), Vec2::new(5.0,3.0));
    for x in line.into_iter() {
        println!("{}", x);
    }

    println!("Iter y!");

    let line = triangle_raster::Line::new(Vec2::new(1.0, 0.5), Vec2::new(5.0,3.0));
    for x in line.into_iter_y() {
        println!("{}", x);
    }


    let trig = Triangle::new(Vec2::new(5.0,100.0), Vec2::new(2.0,50.0), Vec2::new(20.0,10.0));
    println!("Triangle: {:?}", trig);
    for v in trig.into_iter() {
        println!("{}", v);
    }
    //let mut v = Vec::new();

    let mut buf: ImageBuffer<Rgba<u8>, Vec<u8>> = RgbaImage::new(256, 256);
    buf.fill(255);    
    triangle_raster::iter_triangle(&trig, |x| buf.put_pixel(x.x as u32, x.y as u32, Rgba([0,0,0,255])));
    let image = DynamicImage::ImageRgba8(buf);
    image.save("t3st.png");
    //println!("{:?}", v);

    //return;
    let aabb2 = Aabb::new(Vec3f::new(2.0,0.0,0.0), Vec3f::new(1.0, 1.0, 1.5))
       .color(Rgba([255,255,255,255]));
    let grad = Gradient::new(Vec3f::new(1.9,0.0,0.0), Vec3f::new(2.1,0.0,0.0)
    , Rgba([255,0,0,255]), Rgba([255,255,255,255]), Rc::new(aabb2.into()));
    let sphere = Sphere::new(Vec3f::new(-2.0,0.0,0.0), 2.0).color(Rgba([255,0,0,255]));
    let grad2 = Gradient::new(Vec3f::new(0.0,-0.2 + 0.2,0.0), Vec3f::new(0.0,0.2+ 0.2,0.0)
    , Rgba([255,255,255,0]), Rgba([0,0,255,255]), Rc::new(sphere.into()));

    let grad3 =  Gradient::new(Vec3f::new(-2.2, 0.0, 0.0), Vec3f::new(-1.8,0.0,0.0), 
            Rgba([0,255,0,255]), 
            Rgba([0,0,255,0]), 
            Rc::new(grad2.into()));
    let noise = Noise::new(123, Rgba([95,155,55,255]), 
        Rgba([0,0,255,0]), Rc::new(grad3.into()));

    let sdf = DistanceFieldEnum::Empty{}.Insert2(noise) 
        //.Insert2(Sphere::new(Vec3f::new(2.0,0.0,0.0), 2.0).color(Rgba([0, 255,0,255])))
        //.Insert2(Sphere::new(Vec3f::new(0.0,2.0,0.0), 2.0).color(Rgba([255, 255,0,255])))
        //.Insert2(Sphere::new(Vec3f::new(0.0,-2.0,0.0), 2.0).color(Rgba([255, 255,255,255])))
        //.Insert2(Sphere::new(Vec3f::new(0.0,0.0,2.0), 2.0).color(Rgba([255, 0,255,255])))
        //.Insert2(Sphere::new(Vec3f::new(0.0,0.0,-2.0), 2.0).color(Rgba([0, 0,255,255])))
        
        .Insert2(grad)
        ;

    let d = sdf.distance(Vec3f::new(6.5, 5.0, 0.0));
    println!("distance: {}", d);

    let sdf2 = sdf.optimize_bounds();
    let d2 = sdf2.distance(Vec3f::new(6.5, 5.0, 0.0));
    

    println!("distance: {}", d2);
    println!("{:?} \n\n {:?}", sdf, sdf2);

    let mut r = VertexesList::new();
    
    marching_cubes_sdf(&mut r, &sdf2, Vec3f::zeros(), 5.0, 0.2);
    
    let mut window = Window::new("Kiss3d: wasm example");

    let meshtex = r.to_mesh(&sdf);
    let mut mesh = meshtex.0;
    let tex = meshtex.1;
    tex.save("test.png");
    
    let mut c = window.add_mesh(Rc::new(RefCell::new(mesh)), Vec3f::new(0.2, 0.2, 0.2));//window.add_cube(0.5, 0.5, 0.5);
    let mut tm = TextureManager::new();

    c.append_translation(&Translation3::new(0.0, 0.0, 1.0));
    
    let mut tex2 = tm.add_image(tex, "Hello");
    
    c.set_texture(tex2);
    

    c.enable_backface_culling(true);
    window.set_light(Light::StickToCamera);
    
    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.000);
    let state = AppState { c, rot };

    window.render_loop(state);
}
