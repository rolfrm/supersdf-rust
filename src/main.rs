mod mc;
mod sdf;
mod sdf_mesh;
mod sdf_scene;
mod triangle_raster;
mod app_state;
mod vec3;
mod vec2;
mod csg;
use sdf::*;
use sdf_scene::*;
use app_state::*;
use kiss3d::light::Light; 
use vec3::Vec3;


use kiss3d::window::{Window};

fn main() {
    
    let sdf = build_test().optimize_bounds();
    println!("Final sdf: {:?}", sdf);
    let sdf_iterator = SdfScene::new(sdf).with_eye_pos(Vec3::new(58.0, -31.0, -12.0));

    let mut window = Window::new("Kiss3d: wasm example");
    
    window.set_light(Light::StickToCamera);

    let state = AppState::new(sdf_iterator);

    window.render_loop(state);
}
