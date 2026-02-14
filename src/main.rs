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
use glfw::{Action, Context, Key, MouseButton};
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

fn compile_gl_shader(src: &str, shader_type: GLenum) -> GLuint {
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
            eprintln!("Shader compilation failed:\n{}", msg);
            gl::DeleteShader(shader);
            return 0;
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
            eprintln!("Program linking failed:\n{}", msg);
            gl::DeleteProgram(program);
            gl::DeleteShader(vs);
            gl::DeleteShader(fs);
            return 0;
        }

        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
        program
    }
}

struct ShaderProgram {
    id: GLuint,
    u_resolution: GLint,
    u_camera_pos: GLint,
    u_camera_dir: GLint,
    u_camera_up: GLint,
}

/// Compile an SDF tree into a GL shader program. Returns None on failure.
fn build_sdf_program(sdf: &DistanceFieldEnum) -> Option<ShaderProgram> {
    let frag_src = sdf_compiler::compile_sdf_shader(sdf);
    println!("--- Fragment Shader ---\n{}\n--- End Shader ---", frag_src);

    let vs = compile_gl_shader(VERTEX_SHADER_SRC, gl::VERTEX_SHADER);
    if vs == 0 { return None; }

    let fs = compile_gl_shader(&frag_src, gl::FRAGMENT_SHADER);
    if fs == 0 {
        unsafe { gl::DeleteShader(vs); }
        return None;
    }

    let id = link_program(vs, fs);
    if id == 0 { return None; }

    unsafe {
        gl::UseProgram(id);
        Some(ShaderProgram {
            id,
            u_resolution: gl::GetUniformLocation(id, CString::new("u_resolution").unwrap().as_ptr()),
            u_camera_pos: gl::GetUniformLocation(id, CString::new("u_camera_pos").unwrap().as_ptr()),
            u_camera_dir: gl::GetUniformLocation(id, CString::new("u_camera_dir").unwrap().as_ptr()),
            u_camera_up: gl::GetUniformLocation(id, CString::new("u_camera_up").unwrap().as_ptr()),
        })
    }
}

fn build_initial_scene() -> DistanceFieldEnum {
    let s1: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 10.0).into();
    let s1 = s1.colored(Color::rgb(1.0, 0.0, 0.0));

    let s2: DistanceFieldEnum = Sphere::new(Vec3::new(50.0, 0.0, 0.0), 10.0).into();
    let s2 = s2.colored(Color::rgb(0.0, 1.0, 0.0));
    let sdf: DistanceFieldEnum = Add::new(s1, s2).into();

    let s3: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 50.0), 10.0).into();
    let s3 = s3.colored(Color::rgb(0.0, 1.0, 1.0));
    let sdf: DistanceFieldEnum = Add::new(sdf, s3).into();

    let s4: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 50.0, 0.0), 10.0).into();
    let s4 = s4.colored(Color::rgb(1.0, 1.0, 0.0));
    let sdf: DistanceFieldEnum = Add::new(sdf, s4).into();

    let s5: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, -50.0, 0.0), 10.0).into();
    let s5 = s5.colored(Color::rgb(1.0, 0.0, 1.0));
    let sdf: DistanceFieldEnum = Add::new(sdf, s5).into();

    let sub = Vec3::new(12.840058, 74.62816, 8.423447);
    let sdf = sdf.subtract(DistanceFieldEnum::sphere(sub, 2.0));

    return sdf.optimize_bounds()
}

fn main() {
    let mut sdf = build_initial_scene();
    let mut sdf_dirty = true;
    let mut sphere_count: u32 = 0;

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
    window.set_mouse_button_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_cursor_pos_polling(true);
    window.make_current();

    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

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

    // Initial shader program (will be built on first frame via dirty flag)
    let mut program: Option<ShaderProgram> = None;

    // Camera state
    let mut cam_pos = Vec3::new(0.0, 0.0, -60.0);
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.0;
    let mut last_cursor = (0.0f64, 0.0f64);
    let mut first_mouse = true;
    let move_speed = 1.0f32;
    let mouse_sensitivity = 0.002f32;

    // Track key held state via events (more reliable than get_key polling)
    let mut key_w = false;
    let mut key_a = false;
    let mut key_s = false;
    let mut key_d = false;

    while !window.should_close() {
        glfw.poll_events();

        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(key, _, action, _) => {
                    let pressed = action != Action::Release;
                    match key {
                        Key::Escape if pressed => window.set_should_close(true),
                        Key::W => key_w = pressed,
                        Key::A => key_a = pressed,
                        Key::S => key_s = pressed,
                        Key::D => key_d = pressed,
                        Key::Space if pressed => {
                            sphere_count += 1;
                            let new_sphere = Sphere::new(cam_pos, 5.0)
                                .color(Color::rgb(
                                    (sphere_count * 97 % 255) as f32 / 255.0,
                                    (sphere_count * 53 % 255) as f32 / 255.0,
                                    (sphere_count * 179 % 255) as f32 / 255.0,
                                ));
                            sdf = sdf.add(new_sphere).optimize_bounds();
                            sdf_dirty = true;
                            println!("Added sphere #{} at {}", sphere_count, cam_pos);
                        }
                        Key::R if pressed => {
                            sdf = build_initial_scene();
                            sphere_count = 0;
                            sdf_dirty = true;
                            println!("Reset scene");
                        }
                        _ => {}
                    }
                }
                glfw::WindowEvent::MouseButton(MouseButton::Button2, Action::Press, _) => {
                    // Right-click: cast ray through cursor, subtract sphere at hit
                    let (win_w, win_h) = window.get_framebuffer_size();
                    let (cx, cy) = (win_w as f32 / 2.0, win_h as f32 / 2.0);
                    let ux = (cx as f32 - 0.5 * win_w as f32) / win_h as f32;
                    let uy = -(cy as f32 - 0.5 * win_h as f32) / win_h as f32;
                    let cam_dir = Vec3::new(
                        yaw.sin() * pitch.cos(),
                        pitch.sin(),
                        yaw.cos() * pitch.cos(),
                    ).normalize();
                    let cam_up = Vec3::new(0.0, 1.0, 0.0);
                    let cam_right = cam_up.cross(cam_dir).normalize();
                    let cam_up2 = cam_dir.cross(cam_right);
                    let ray_dir = (cam_right * ux + cam_up2 * uy + cam_dir).normalize();
                    

                    if let Some((_dist, hit_pos)) = sdf.cast_ray(cam_pos, ray_dir, 1000.0) {
                        sdf = sdf.subtract(DistanceFieldEnum::sphere(hit_pos, 5.0));
                        sdf = sdf.optimize_bounds();
                        sdf_dirty = true;
                        println!("Subtracted sphere at {}", hit_pos);
                    }
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

        // Recompile shader if SDF changed
        if sdf_dirty {
            sdf_dirty = false;
            if let Some(old) = &program {
                unsafe { gl::DeleteProgram(old.id); }
            }
            match build_sdf_program(&sdf) {
                Some(p) => {
                    println!("Shader recompiled successfully");
                    program = Some(p);
                }
                None => {
                    eprintln!("Failed to compile shader for current SDF");
                    program = None;
                }
            }
        }

        // Camera direction from yaw/pitch
        let dir = Vec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            yaw.cos() * pitch.cos(),
        ).normalize();
        let up = Vec3::new(0.0, 1.0, 0.0);
        let right = up.cross(dir).normalize();

        // WASD movement
        if key_w {
            cam_pos = cam_pos + dir * move_speed;
        }
        if key_s {
            cam_pos = cam_pos - dir * move_speed;
        }
        if key_a {
            cam_pos = cam_pos - right * move_speed;
        }
        if key_d {
            cam_pos = cam_pos + right * move_speed;
        }

        let (w, h) = window.get_framebuffer_size();

        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            if let Some(prog) = &program {
                gl::UseProgram(prog.id);
                gl::Uniform2f(prog.u_resolution, w as f32, h as f32);
                gl::Uniform3f(prog.u_camera_pos, cam_pos.x, cam_pos.y, cam_pos.z);
                gl::Uniform3f(prog.u_camera_dir, dir.x, dir.y, dir.z);
                gl::Uniform3f(prog.u_camera_up, up.x, up.y, up.z);

                gl::BindVertexArray(vao);
                gl::DrawArrays(gl::TRIANGLES, 0, 6);
            }
        }

        window.swap_buffers();
    }

    if let Some(prog) = &program {
        unsafe { gl::DeleteProgram(prog.id); }
    }
    unsafe {
        gl::DeleteVertexArrays(1, &vao);
        gl::DeleteBuffers(1, &vbo);
    }
}
