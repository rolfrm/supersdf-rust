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
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::rc::Rc;
use std::str;

// ---------- Grid constants ----------
const GRID_N: usize = 80;
const GRID_MIN: f32 = -200.0;
const BLOCK_SIZE: f32 = 5.0;

// ---------- Block vertex shader ----------
const BLOCK_VERTEX_SHADER_SRC: &str = r#"
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 u_model;
uniform mat4 u_vp;
out vec3 v_world_pos;
void main() {
    vec4 world = u_model * vec4(aPos, 1.0);
    v_world_pos = world.xyz;
    gl_Position = u_vp * world;
}
"#;

// ---------- GL helpers ----------

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
            eprintln!("Shader: {}", src);
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

// ---------- Mat4 helpers (column-major) ----------

type Mat4 = [f32; 16];

fn mat4_perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fovy / 2.0).tan();
    let nf = near - far;
    [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) / nf, -1.0,
        0.0, 0.0, 2.0 * far * near / nf, 0.0,
    ]
}

fn mat4_view(eye: Vec3, dir: Vec3, world_up: Vec3) -> Mat4 {
    let right = world_up.cross(dir).normalize();
    let up = dir.cross(right);
    let tx = right.dot(eye);
    let ty = up.dot(eye);
    let tz = dir.dot(eye);
    [
        right.x, up.x, -dir.x, 0.0,
        right.y, up.y, -dir.y, 0.0,
        right.z, up.z, -dir.z, 0.0,
        -tx, -ty, tz, 1.0,
    ]
}

fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

fn mat4_block_model(block_min: Vec3, size: f32) -> Mat4 {
    [
        size, 0.0, 0.0, 0.0,
        0.0, size, 0.0, 0.0,
        0.0, 0.0, size, 0.0,
        block_min.x, block_min.y, block_min.z, 1.0,
    ]
}

// ---------- Block program ----------

struct BlockProgram {
    id: GLuint,
    u_vp: GLint,
    u_model: GLint,
    u_camera_pos: GLint,
    u_block_min: GLint,
    u_block_max: GLint,
    u_params: GLint,
}

fn build_block_program(frag_src: &str) -> Option<BlockProgram> {
    println!("compile: {}", frag_src);
    let vs = compile_gl_shader(BLOCK_VERTEX_SHADER_SRC, gl::VERTEX_SHADER);
    if vs == 0 { return None; }

    let fs = compile_gl_shader(frag_src, gl::FRAGMENT_SHADER);
    if fs == 0 {
        unsafe { gl::DeleteShader(vs); }
        return None;
    }

    let id = link_program(vs, fs);
    if id == 0 { return None; }

    unsafe {
        Some(BlockProgram {
            id,
            u_vp: gl::GetUniformLocation(id, CString::new("u_vp").unwrap().as_ptr()),
            u_model: gl::GetUniformLocation(id, CString::new("u_model").unwrap().as_ptr()),
            u_camera_pos: gl::GetUniformLocation(id, CString::new("u_camera_pos").unwrap().as_ptr()),
            u_block_min: gl::GetUniformLocation(id, CString::new("u_block_min").unwrap().as_ptr()),
            u_block_max: gl::GetUniformLocation(id, CString::new("u_block_max").unwrap().as_ptr()),
            u_params: gl::GetUniformLocation(id, CString::new("u_params").unwrap().as_ptr()),
        })
    }
}

// ---------- Grid ----------

struct Grid {
    slot_hashes: Vec<u64>,
    slot_params: Vec<Vec<f32>>,
    programs: HashMap<u64, BlockProgram>,
}

impl Grid {
    fn new() -> Grid {
        let n = GRID_N * GRID_N * GRID_N;
        Grid {
            slot_hashes: vec![0u64; n],
            slot_params: vec![Vec::new(); n],
            programs: HashMap::new(),
        }
    }

    fn rebuild(&mut self, sdf: &DistanceFieldEnum) {
        let mut cache = HashSet::new();
        let n = GRID_N * GRID_N * GRID_N;
        let mut new_hashes = vec![0u64; n];
        let mut new_params = vec![Vec::new(); n];
        let mut to_compile: HashMap<u64, Rc<DistanceFieldEnum>> = HashMap::new();
        let mut empty_count = 0u32;

        // Phase 1: optimize all blocks, collect unique topology hashes
        for ix in 0..GRID_N {
            for iy in 0..GRID_N {
                for iz in 0..GRID_N {
                    let idx = ix * GRID_N * GRID_N + iy * GRID_N + iz;
                    let center = Vec3::new(
                        GRID_MIN + (ix as f32 + 0.5) * BLOCK_SIZE,
                        GRID_MIN + (iy as f32 + 0.5) * BLOCK_SIZE,
                        GRID_MIN + (iz as f32 + 0.5) * BLOCK_SIZE,
                    );
                    let optimized = sdf.optimized_for_block(center, BLOCK_SIZE, &mut cache);
                    let half_diag = BLOCK_SIZE * 0.5 * 1.7320508;
                    let block_empty = matches!(optimized.as_ref(), DistanceFieldEnum::Empty)
                        || optimized.distance(center) > half_diag;

                    if block_empty {
                        empty_count += 1;
                        continue;
                    }

                    let hash = optimized.topology_hash();
                    new_hashes[idx] = hash;
                    new_params[idx] = sdf_compiler::collect_block_sdf_params(&optimized);
                    to_compile.entry(hash).or_insert(optimized);
                }
            }
        }

        // Phase 2: delete programs no longer needed
        let needed: HashSet<u64> = new_hashes.iter().copied().filter(|&h| h != 0).collect();
        self.programs.retain(|hash, prog| {
            if needed.contains(hash) {
                true
            } else {
                unsafe { gl::DeleteProgram(prog.id); }
                false
            }
        });

        // Phase 3: compile only new unique programs
        let mut compiled = 0u32;
        let mut reused = 0u32;
        for (hash, optimized) in &to_compile {
            if self.programs.contains_key(hash) {
                reused += 1;
            } else {
                let compiled_shader = sdf_compiler::compile_block_sdf_shader(optimized);
                match build_block_program(&compiled_shader.source) {
                    Some(prog) => {
                        self.programs.insert(*hash, prog);
                        compiled += 1;
                    }
                    None => {
                        eprintln!("Failed to compile shader (hash {})", hash);
                    }
                }
            }
        }

        self.slot_hashes = new_hashes;
        self.slot_params = new_params;
        let active = self.slot_hashes.iter().filter(|&&h| h != 0).count();
        println!("Grid rebuild: {} active blocks, {} unique ({} compiled, {} reused), {} empty",
            active, to_compile.len(), compiled, reused, empty_count);
    }

    fn cleanup(&mut self) {
        for (_, prog) in self.programs.drain() {
            unsafe { gl::DeleteProgram(prog.id); }
        }
    }
}

// ---------- Scene ----------

fn build_initial_scene() -> DistanceFieldEnum {
    
    let mut sdf: DistanceFieldEnum = DistanceFieldEnum::Empty;
    /*   

        for i in 0..8 {
            let offset = (i - 4) as f32 * 50.0;
            let s2: DistanceFieldEnum = Sphere::new(Vec3::new(20.0 + offset, 0.0, 0.0), 10.0).into();
            let s2 = s2.colored(Color::rgb(0.0, 1.0, 0.0));
            sdf  = Add::new(sdf, s2).into();
            
            let s3: DistanceFieldEnum = Sphere::new(Vec3::new(0.0 + offset, 0.0, 20.0), 10.0).into();
            let s3 = s3.colored(Color::rgb(0.0, 1.0, 1.0));
             sdf = Add::new(sdf, s3).into();
            
            let s4: DistanceFieldEnum = Sphere::new(Vec3::new(0.0 + offset, 20.0, 0.0), 10.0).into();
            let s4 = s4.colored(Color::rgb(1.0, 1.0, 0.0));
            sdf = Add::new(sdf, s4).into();
            
            let s5: DistanceFieldEnum = Sphere::new(Vec3::new(0.0 + offset, -20.0, 0.0), 10.0).into();
            let s5 = s5.colored(Color::rgb(1.0, 0.0, 1.0));
            sdf = Add::new(sdf, s5).into();
            
            let sub = Vec3::new(12.840058, 74.62816, 8.423447);
            sdf = sdf.subtract(DistanceFieldEnum::sphere(sub, 2.0));


}
     */
    for i in (-60..60).step_by(20) {
        for j in (-60..60).step_by(20) {
            for h in (-60..60).step_by(20) {
    sdf = Add::new(sdf, DistanceFieldEnum::sphere(Vec3::new(i as f32, j as f32, h as f32), 5.0).colored(Color::rgb(0.0, 1.0, 0.0))).into();
    }}}
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

    // Unit cube [0,1]^3 - 36 vertices (12 triangles)
    let vertices: [f32; 108] = [
        // Front (z=1)
        0.0,0.0,1.0, 1.0,0.0,1.0, 1.0,1.0,1.0,
        0.0,0.0,1.0, 1.0,1.0,1.0, 0.0,1.0,1.0,
        // Back (z=0)
        1.0,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0,
        1.0,0.0,0.0, 0.0,1.0,0.0, 1.0,1.0,0.0,
        // Left (x=0)
        0.0,0.0,0.0, 0.0,0.0,1.0, 0.0,1.0,1.0,
        0.0,0.0,0.0, 0.0,1.0,1.0, 0.0,1.0,0.0,
        // Right (x=1)
        1.0,0.0,1.0, 1.0,0.0,0.0, 1.0,1.0,0.0,
        1.0,0.0,1.0, 1.0,1.0,0.0, 1.0,1.0,1.0,
        // Top (y=1)
        0.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0,0.0,
        0.0,1.0,1.0, 1.0,1.0,0.0, 0.0,1.0,0.0,
        // Bottom (y=0)
        0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,0.0,1.0,
        0.0,0.0,0.0, 1.0,0.0,1.0, 0.0,0.0,1.0,
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
            0, 3, gl::FLOAT, gl::FALSE,
            (3 * std::mem::size_of::<f32>()) as GLsizei,
            ptr::null(),
        );
        gl::EnableVertexAttribArray(0);
        gl::BindVertexArray(0);
    }

    let mut grid = Grid::new();

    // Camera state
    let mut cam_pos = Vec3::new(0.0, 0.0, -60.0);
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.0;
    let mut last_cursor = (0.0f64, 0.0f64);
    let mut first_mouse = true;
    let move_speed = 1.0f32;
    let mouse_sensitivity = 0.002f32;

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

        // Recompile shaders if SDF changed
        if sdf_dirty {
            sdf_dirty = false;
            grid.rebuild(&sdf);
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
        let aspect = w as f32 / h as f32;

        // Build VP matrix: FOV matches old shader's focal length of 1.0
        let fovy = 2.0 * (0.5f32).atan();
        let view = mat4_view(cam_pos, dir, up);
        let proj = mat4_perspective(fovy, aspect, 0.1, 500.0);
        let vp = mat4_mul(&proj, &view);

        unsafe {
            // Sky background
            gl::ClearColor(0.25, 0.35, 0.55, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::DEPTH_TEST);
            gl::Disable(gl::CULL_FACE);

            gl::BindVertexArray(vao);

            for ix in 0..GRID_N {
                for iy in 0..GRID_N {
                    for iz in 0..GRID_N {
                        let idx = ix * GRID_N * GRID_N + iy * GRID_N + iz;
                        let hash = grid.slot_hashes[idx];
                        if hash == 0 { continue; }

                        if let Some(prog) = grid.programs.get(&hash) {
                            let block_min = Vec3::new(
                                GRID_MIN + ix as f32 * BLOCK_SIZE,
                                GRID_MIN + iy as f32 * BLOCK_SIZE,
                                GRID_MIN + iz as f32 * BLOCK_SIZE,
                            );
                            let block_max = block_min + Vec3::new(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
                            let model = mat4_block_model(block_min, BLOCK_SIZE);

                            gl::UseProgram(prog.id);
                            gl::UniformMatrix4fv(prog.u_vp, 1, gl::FALSE, vp.as_ptr());
                            gl::UniformMatrix4fv(prog.u_model, 1, gl::FALSE, model.as_ptr());
                            gl::Uniform3f(prog.u_camera_pos, cam_pos.x, cam_pos.y, cam_pos.z);
                            gl::Uniform3f(prog.u_block_min, block_min.x, block_min.y, block_min.z);
                            gl::Uniform3f(prog.u_block_max, block_max.x, block_max.y, block_max.z);

                            let params = &grid.slot_params[idx];
                            if !params.is_empty() {
                                gl::Uniform1fv(prog.u_params, params.len() as i32, params.as_ptr());
                            }

                            gl::DrawArrays(gl::TRIANGLES, 0, 36);
                        }
                    }
                }
            }
        }

        window.swap_buffers();
    }

    grid.cleanup();
    unsafe {
        gl::DeleteVertexArrays(1, &vao);
        gl::DeleteBuffers(1, &vbo);
    }
}
