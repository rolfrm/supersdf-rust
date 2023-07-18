mod mc;
mod sdf;
mod sdf_mesh;
mod sdf_scene;
mod triangle_raster;
mod app_state;

use sdf::*;
use sdf_scene::*;
use app_state::*;


use kiss3d::light::Light;


use image::{Rgba};

use kiss3d::nalgebra::{
    Vector2, Vector3
};
use kiss3d::window::{Window};

type Vec3f = Vector3<f32>;
type Vec3 = Vec3f;
type Vec2 = Vector2<f32>;



fn main() {
    let sdf : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 2.0)
        .color(Rgba([255, 0, 0, 255]))
        .into();// build_test();
    let sdf = build_test();
    let sdf2 = sdf.optimize_bounds();

    let sdf_iterator = SdfScene {
        sdf: sdf2,
        eye_pos: Vec3::zeros(),
        block_size: 5.0,
        render_blocks: Vec::new(),
    };

    let mut window = Window::new("Kiss3d: wasm example");
    

    window.set_light(Light::StickToCamera);

    let state = AppState::new(sdf_iterator);

    window.render_loop(state);
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_optimize_bounds() {
        let sdf = build_test();
        let sdf2 = sdf.optimize_bounds();
    
        let mut sdf_iterator = SdfScene {
            sdf: sdf2,
            eye_pos: Vec3::zeros(),
            block_size: 2.0,
            render_blocks: Vec::new(),
        };

        sdf_iterator.iterate_scene(Vec3::new(0.0, 0.0, 0.0), 64.0);
        for block in sdf_iterator.render_blocks {
            println!("{:?}", block.2);
            
        }
    }



}
