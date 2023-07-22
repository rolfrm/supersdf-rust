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

use kiss3d::window::{Window};

fn main() {
    let sdf = build_test2().optimize_bounds();

    let sdf_iterator = SdfScene::new(sdf);

    let mut window = Window::new("Kiss3d: wasm example");
    
    window.set_light(Light::StickToCamera);

    let state = AppState::new(sdf_iterator);

    window.render_loop(state);
}
