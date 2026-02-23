pub mod sdf;
pub mod vec3;
mod vec2;
pub mod color;
pub mod mat4;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn main() -> Result<(), JsValue> {
    Ok(())
}
