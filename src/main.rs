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
use vec3::Vec3;

extern crate glfw;
extern crate gl;

use glfw::{Action, Context, Key, MouseButton};

fn main() {
    let s2: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(1.0, 0.0, 0.0));
    let a1 = s2;

    let s2: DistanceFieldEnum = Sphere::new(Vec3::new(50.0, 0.0, 0.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(0.0, 1.0, 0.0));
    let a1: DistanceFieldEnum = Add::new(a1, s2).into();

    let s2: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 50.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(0.0, 1.0, 1.0));
    let a1: DistanceFieldEnum = Add::new(a1, s2).into();

    let s2: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 50.0, 0.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(1.0, 1.0, 0.0));
    let a1: DistanceFieldEnum = Add::new(a1, s2).into();

    let s2: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, -50.0, 0.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(1.0, 0.0, 1.0));
    let a1: DistanceFieldEnum = Add::new(a1, s2).into();

    let sub = Vec3::new(12.840058, 74.62816, 8.423447);
    let a1 = a1.subtract(DistanceFieldEnum::sphere(sub, 2.0));

    let sdf = a1.optimize_bounds();
    println!("Final sdf: {:?}", sdf);
    let sdf_iterator = SdfScene::new(sdf).with_eye_pos(Vec3::new(10.0, 60.0, 0.0));

    // Initialize GLFW
    let mut glfw = glfw::init(glfw::fail_on_errors).expect("Failed to init GLFW");

    // Request OpenGL 2.1 context (legacy, macOS compatible)
    glfw.window_hint(glfw::WindowHint::ContextVersion(2, 1));

    let (mut window, events) = glfw
        .create_window(1024, 768, "SuperSDF", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window");

    window.make_current();
    window.set_key_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_mouse_button_polling(true);
    window.set_framebuffer_size_polling(true);

    // Load OpenGL function pointers
    gl::load_with(|s| window.get_proc_address(s) as *const _);

    unsafe {
        gl::ClearColor(0.1, 0.1, 0.15, 1.0);
        gl::Enable(gl::DEPTH_TEST);
    }

    let mut state = AppState::new(sdf_iterator);

    let mut last_cursor = (0.0f64, 0.0f64);
    let mut mouse_captured = false;
    let mut first_mouse = true;

    while !window.should_close() {
        glfw.poll_events();

        let (w, h) = window.get_framebuffer_size();
        unsafe {
            gl::Viewport(0, 0, w, h);
        }

        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    if mouse_captured {
                        mouse_captured = false;
                        window.set_cursor_mode(glfw::CursorMode::Normal);
                    } else {
                        window.set_should_close(true);
                    }
                }
                glfw::WindowEvent::Key(Key::Enter, _, Action::Press, _) => {
                    state.clear_cache();
                }
                glfw::WindowEvent::Key(Key::F, _, Action::Press, _) => {
                    state.toggle_wireframe();
                }
                glfw::WindowEvent::Key(Key::V, _, Action::Press, _) => {
                    state.toggle_voxels();
                }
                glfw::WindowEvent::MouseButton(MouseButton::Button1, Action::Press, _) => {
                    if !mouse_captured {
                        mouse_captured = true;
                        first_mouse = true;
                        window.set_cursor_mode(glfw::CursorMode::Disabled);
                    }
                }
                glfw::WindowEvent::MouseButton(MouseButton::Button2, Action::Press, _) => {
                    let origin = state.camera.position;
                    let dir = state.camera.front();
                    state.handle_right_click(origin, dir);
                }
                glfw::WindowEvent::CursorPos(x, y) => {
                    if mouse_captured {
                        if first_mouse {
                            last_cursor = (x, y);
                            first_mouse = false;
                        }
                        let dx = (x - last_cursor.0) as f32;
                        let dy = (y - last_cursor.1) as f32;
                        state.camera.rotate(dx, dy);
                    }
                    last_cursor = (x, y);
                }
                _ => {}
            }
        }

        // Continuous key input for movement
        if mouse_captured {
            if window.get_key(Key::W) == Action::Press {
                state.camera.move_forward(1.0);
            }
            if window.get_key(Key::S) == Action::Press {
                state.camera.move_backward(1.0);
            }
            if window.get_key(Key::A) == Action::Press {
                state.camera.move_left(1.0);
            }
            if window.get_key(Key::D) == Action::Press {
                state.camera.move_right(1.0);
            }
        }

        state.step(w as u32, h as u32);

        window.swap_buffers();
    }
}
