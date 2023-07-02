pub mod mc;
mod sdf;
mod sdf_mesh;
use std::cell::RefCell;
use std::rc::Rc;

use kiss3d::resource::{TextureManager, MaterialManager, Material};
use sdf::*;
use sdf_mesh::*;

use kiss3d::light::Light;
use kiss3d::scene::SceneNode;

use kiss3d::window::{State, Window};
use kiss3d::nalgebra::{UnitQuaternion, Vector3, Point3, Translation3, Vector2};
use image::{Rgba};

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
    let aabb2 = Aabb::new(Vec3f::new(2.0,0.0,0.0), Vec3f::new(1.0, 1.0, 1.5))
       .color(Rgba([255,255,255,255]));
    let grad = Gradient::new(Vec3f::new(1.0,0.0,0.0), Vec3f::new(3.0,0.0,0.0)
    , Rgba([255,0,0,255]), Rgba([255,255,255,255]), Rc::new(aabb2.into()));
    let sphere = Sphere::new(Vec3f::new(-2.0,0.0,0.0), 2.0);
    let grad2 = Gradient::new(Vec3f::new(0.0,-0.2,0.0), Vec3f::new(0.0,0.2,0.0)
    , Rgba([255,255,255,255]), Rgba([0,0,255,255]), Rc::new(sphere.into()));
    

    let sdf = DistanceFieldEnum::Empty{}.Insert2(grad2) 
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
    
    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.005);
    let state = AppState { c, rot };

    window.render_loop(state);
}
