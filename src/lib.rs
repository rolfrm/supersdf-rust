mod mc;
mod sdf;
mod sdf_mesh;
mod sdf_scene;
mod triangle_raster;
mod app_state;
mod vec3;

use sdf::*;
use sdf_scene::*;
use app_state::*;

use kiss3d::light::Light;

use kiss3d::window::{Window};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn main() -> Result<(), JsValue> {
    let sdf = build_test().optimize_bounds();

    let sdf_iterator = SdfScene::new(sdf);

    let mut window = Window::new("Kiss3d: wasm example");
    
    window.set_light(Light::StickToCamera);

    let state = AppState::new(sdf_iterator);


    window.render_loop(state);
    Ok(())
}
