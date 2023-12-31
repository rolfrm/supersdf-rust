mod mc;
mod sdf;
mod sdf_mesh;
mod sdf_scene;
mod triangle_raster;
mod app_state;
mod vec3;
mod vec2;
mod color;
mod csg;
use color::*;
use sdf::*;
use sdf_scene::*;
use app_state::*;
use kiss3d::light::Light; 
use vec3::Vec3;


use kiss3d::window::{Window};

fn main() {
    
    let s2 : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(1.0,0.0,0.0));
    let a1 = s2;
    
    let s2 : DistanceFieldEnum = Sphere::new(Vec3::new(50.0, 0.0, 00.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(0.0,1.0,0.0));
    let a1 : DistanceFieldEnum = Add::new(a1,s2).into();

    let s2 : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 50.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(0.0,1.0,1.0));
    let a1 : DistanceFieldEnum = Add::new(a1,s2).into();

    let s2 : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 50.0, 00.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(1.0,1.0,0.0));
    let a1 : DistanceFieldEnum = Add::new(a1,s2).into();

    let s2 : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, -50.0, 00.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(1.0,0.0,1.0));
    let a1 : DistanceFieldEnum = Add::new(a1,s2).into();


    let sub = Vec3::new(12.840058, 74.62816, 8.423447);
    let a1 = a1.subtract(DistanceFieldEnum::sphere(sub, 2.0));



    let sdf = a1.optimize_bounds();
    println!("Final sdf: {:?}", sdf);
    let sdf_iterator = SdfScene::new(sdf).with_eye_pos(Vec3::new(10.0, 60.0, 0.0));

    let mut window = Window::new("Kiss3d: wasm example");
    
    window.set_light(Light::StickToCamera);

    let state = AppState::new(sdf_iterator);

    window.render_loop(state);
}
