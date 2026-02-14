extern crate gl;
extern crate glfw;

mod sdf;
mod vec3;
mod vec2;
mod color;
mod sdf_compiler;

use color::*;
use sdf::*;
use vec3::Vec3;

use gl::types::*;
use glfw::{Action, Context, Key};
use std::ffi::CString;
use std::ptr;
use std::str;

const VERTEX_SHADER_SRC: &str = r#"
#version 330 core
layout (location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"#;

fn compile_shader(src: &str, shader_type: GLenum) -> GLuint {
    unsafe {
        let shader = gl::CreateShader(shader_type);
        let c_str = CString::new(src.as_bytes()).unwrap();
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null());
        gl::CompileShader(shader);

        let mut success = gl::FALSE as GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = vec![0u8; len as usize];
            gl::GetShaderInfoLog(shader, len, ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
            let msg = str::from_utf8(&buf).unwrap_or("(invalid utf8)");
            panic!("Shader compilation failed:\n{}", msg);
        }
        shader
    }
}

fn link_program(vs: GLuint, fs: GLuint) -> GLuint {
    unsafe {
        let program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);

        let mut success = gl::FALSE as GLint;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            let mut len = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = vec![0u8; len as usize];
            gl::GetProgramInfoLog(program, len, ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
            let msg = str::from_utf8(&buf).unwrap_or("(invalid utf8)");
            panic!("Program linking failed:\n{}", msg);
        }

        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
        program
    }
}

fn main() {
    // Build the SDF scene
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

    // Compile SDF -> GLSL
    let frag_src = sdf_compiler::compile_sdf_shader(&sdf);
    println!("--- Generated fragment shader ---");
    println!("{}", frag_src);

    // Init GLFW
    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    #[cfg(target_os = "macos")]
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));

    let (mut window, events) = glfw
        .create_window(1024, 768, "SuperSDF", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window");

    window.set_key_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_cursor_pos_polling(true);
    window.make_current();

    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    // Compile shaders and link program
    let vs = compile_shader(VERTEX_SHADER_SRC, gl::VERTEX_SHADER);
    let fs = compile_shader(&frag_src, gl::FRAGMENT_SHADER);
    let program = link_program(vs, fs);

    // Fullscreen quad (two triangles covering clip space)
    let vertices: [f32; 12] = [
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0,
    ];

    let (mut vao, mut vbo) = (0u32, 0u32);
    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);

        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (vertices.len() * std::mem::size_of::<f32>()) as GLsizeiptr,
            vertices.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::VertexAttribPointer(
            0, 2, gl::FLOAT, gl::FALSE,
            (2 * std::mem::size_of::<f32>()) as GLsizei,
            ptr::null(),
        );
        gl::EnableVertexAttribArray(0);
        gl::BindVertexArray(0);
    }

    // Get uniform locations
    let u_resolution;
    let u_camera_pos;
    let u_camera_dir;
    let u_camera_up;
    unsafe {
        gl::UseProgram(program);
        u_resolution = gl::GetUniformLocation(program, CString::new("u_resolution").unwrap().as_ptr());
        u_camera_pos = gl::GetUniformLocation(program, CString::new("u_camera_pos").unwrap().as_ptr());
        u_camera_dir = gl::GetUniformLocation(program, CString::new("u_camera_dir").unwrap().as_ptr());
        u_camera_up = gl::GetUniformLocation(program, CString::new("u_camera_up").unwrap().as_ptr());
    }

    // Camera state
    let mut cam_pos = Vec3::new(0.0, 0.0, -60.0);
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.0;
    let mut last_cursor = (0.0f64, 0.0f64);
    let mut first_mouse = true;
    let move_speed = 1.0f32;
    let mouse_sensitivity = 0.002f32;

    while !window.should_close() {
        glfw.poll_events();

        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true);
                }
                glfw::WindowEvent::FramebufferSize(w, h) => unsafe {
                    gl::Viewport(0, 0, w, h);
                },
                glfw::WindowEvent::CursorPos(x, y) => {
                    if first_mouse {
                        last_cursor = (x, y);
                        first_mouse = false;
                    }
                    let dx = (x - last_cursor.0) as f32;
                    let dy = (y - last_cursor.1) as f32;
                    last_cursor = (x, y);
                    yaw += dx * mouse_sensitivity;
                    pitch -= dy * mouse_sensitivity;
                    pitch = pitch.clamp(-1.5, 1.5);
                }
                _ => {}
            }
        }

        // Camera direction from yaw/pitch
        let dir = Vec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            yaw.cos() * pitch.cos(),
        ).normalize();
        let up = Vec3::new(0.0, 1.0, 0.0);
        let right = dir.cross(up).normalize();

        // WASD movement
        if window.get_key(Key::W) == Action::Press {
            cam_pos = cam_pos + dir * move_speed;
        }
        if window.get_key(Key::S) == Action::Press {
            cam_pos = cam_pos - dir * move_speed;
        }
        if window.get_key(Key::A) == Action::Press {
            cam_pos = cam_pos - right * move_speed;
        }
        if window.get_key(Key::D) == Action::Press {
            cam_pos = cam_pos + right * move_speed;
        }

        let (w, h) = window.get_framebuffer_size();

        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            gl::UseProgram(program);
            gl::Uniform2f(u_resolution, w as f32, h as f32);
            gl::Uniform3f(u_camera_pos, cam_pos.x, cam_pos.y, cam_pos.z);
            gl::Uniform3f(u_camera_dir, dir.x, dir.y, dir.z);
            gl::Uniform3f(u_camera_up, up.x, up.y, up.z);

            gl::BindVertexArray(vao);
            gl::DrawArrays(gl::TRIANGLES, 0, 6);
        }

        window.swap_buffers();
    }

    unsafe {
        gl::DeleteVertexArrays(1, &vao);
        gl::DeleteBuffers(1, &vbo);
        gl::DeleteProgram(program);
    }
}
