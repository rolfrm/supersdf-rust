mod sdf;
mod vec3;
mod vec2;
mod color;
pub mod sdf_compiler;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn main() -> Result<(), JsValue> {
    Ok(())
}
