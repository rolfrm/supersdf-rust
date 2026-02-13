use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::mem;
use std::ptr;

use gl::types::*;

use crate::{
    sdf_mesh::{marching_cubes_sdf, marching_voxels, RawMesh, TriangleList},
    sdf_scene::{mat4_look_at, mat4_mul, mat4_perspective, Mat4, SdfKey, SdfScene},
    vec3::Vec3,
};
use crate::sdf::*;

// GLSL 120 vertex shader (legacy OpenGL / macOS compatible)
const VERTEX_SHADER_SRC: &str = r#"
#version 120
attribute vec3 aPos;
attribute vec2 aUV;
uniform mat4 uMVP;
varying vec2 vUV;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vUV = aUV;
}
"#;

// GLSL 120 fragment shader
const FRAGMENT_SHADER_SRC: &str = r#"
#version 120
varying vec2 vUV;
uniform sampler2D uTexture;
void main() {
    gl_FragColor = texture2D(uTexture, vUV);
}
"#;

/// Compiled OpenGL shader program
struct ShaderProgram {
    id: GLuint,
    loc_mvp: GLint,
    loc_texture: GLint,
    loc_pos: GLuint,
    loc_uv: GLuint,
}

fn compile_shader(src: &str, kind: GLenum) -> GLuint {
    unsafe {
        let shader = gl::CreateShader(kind);
        let c_str = CString::new(src.as_bytes()).unwrap();
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null());
        gl::CompileShader(shader);

        let mut success = gl::FALSE as GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = vec![0u8; len as usize];
            gl::GetShaderInfoLog(shader, len, ptr::null_mut(), buf.as_mut_ptr() as *mut i8);
            eprintln!("Shader compile error: {}", String::from_utf8_lossy(&buf));
        }
        shader
    }
}

fn create_shader_program() -> ShaderProgram {
    unsafe {
        let vs = compile_shader(VERTEX_SHADER_SRC, gl::VERTEX_SHADER);
        let fs = compile_shader(FRAGMENT_SHADER_SRC, gl::FRAGMENT_SHADER);

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
            gl::GetProgramInfoLog(program, len, ptr::null_mut(), buf.as_mut_ptr() as *mut i8);
            eprintln!("Program link error: {}", String::from_utf8_lossy(&buf));
        }

        gl::DeleteShader(vs);
        gl::DeleteShader(fs);

        let mvp_name = CString::new("uMVP").unwrap();
        let tex_name = CString::new("uTexture").unwrap();
        let pos_name = CString::new("aPos").unwrap();
        let uv_name = CString::new("aUV").unwrap();

        ShaderProgram {
            id: program,
            loc_mvp: gl::GetUniformLocation(program, mvp_name.as_ptr()),
            loc_texture: gl::GetUniformLocation(program, tex_name.as_ptr()),
            loc_pos: gl::GetAttribLocation(program, pos_name.as_ptr()) as GLuint,
            loc_uv: gl::GetAttribLocation(program, uv_name.as_ptr()) as GLuint,
        }
    }
}

/// GPU-side mesh handle
struct GlMesh {
    vbo_pos: GLuint,
    vbo_uv: GLuint,
    ebo: GLuint,
    texture: GLuint,
    num_indices: i32,
}

impl GlMesh {
    fn from_raw(raw: &RawMesh) -> GlMesh {
        unsafe {
            let mut vbo_pos: GLuint = 0;
            let mut vbo_uv: GLuint = 0;
            let mut ebo: GLuint = 0;
            let mut texture: GLuint = 0;

            // positions
            gl::GenBuffers(1, &mut vbo_pos);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo_pos);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (raw.positions.len() * mem::size_of::<f32>()) as GLsizeiptr,
                raw.positions.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            // UVs
            gl::GenBuffers(1, &mut vbo_uv);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo_uv);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (raw.uvs.len() * mem::size_of::<f32>()) as GLsizeiptr,
                raw.uvs.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            // indices
            gl::GenBuffers(1, &mut ebo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                (raw.indices.len() * mem::size_of::<u16>()) as GLsizeiptr,
                raw.indices.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            // texture
            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as GLint);

            let img = raw.image.to_rgba8();
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA as GLint,
                img.width() as GLsizei,
                img.height() as GLsizei,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                img.as_raw().as_ptr() as *const _,
            );

            GlMesh {
                vbo_pos,
                vbo_uv,
                ebo,
                texture,
                num_indices: raw.indices.len() as i32,
            }
        }
    }

    fn destroy(&self) {
        unsafe {
            gl::DeleteBuffers(1, &self.vbo_pos);
            gl::DeleteBuffers(1, &self.vbo_uv);
            gl::DeleteBuffers(1, &self.ebo);
            gl::DeleteTextures(1, &self.texture);
        }
    }
}

/// First-person camera
pub struct FirstPersonCamera {
    pub position: Vec3,
    pub yaw: f32,   // radians, 0 = looking along -Z
    pub pitch: f32,  // radians
    pub speed: f32,
    pub sensitivity: f32,
}

impl FirstPersonCamera {
    pub fn new(pos: Vec3) -> FirstPersonCamera {
        FirstPersonCamera {
            position: pos,
            yaw: 0.0,
            pitch: 0.0,
            speed: 1.0,
            sensitivity: 0.003,
        }
    }

    pub fn front(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }

    pub fn right(&self) -> Vec3 {
        let front = self.front();
        front.cross(Vec3::new(0.0, 1.0, 0.0)).normalize()
    }

    pub fn view_matrix(&self) -> Mat4 {
        let target = self.position + self.front();
        mat4_look_at(self.position, target, Vec3::new(0.0, 1.0, 0.0))
    }

    pub fn move_forward(&mut self, dt: f32) {
        self.position = self.position + self.front() * self.speed * dt;
    }

    pub fn move_backward(&mut self, dt: f32) {
        self.position = self.position - self.front() * self.speed * dt;
    }

    pub fn move_left(&mut self, dt: f32) {
        self.position = self.position - self.right() * self.speed * dt;
    }

    pub fn move_right(&mut self, dt: f32) {
        self.position = self.position + self.right() * self.speed * dt;
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.sensitivity;
        self.pitch -= dy * self.sensitivity;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }
}

pub struct AppState {
    pub sdf_iterator: SdfScene,
    sdf_cache: HashSet<DistanceFieldEnum>,
    nodes: HashMap<SdfKey, (GlMesh, DistanceFieldEnum, f32, Vec3)>,
    pub camera: FirstPersonCamera,
    shader: ShaderProgram,
    time: f32,
    wireframe: bool,
    voxels: bool,
    visible_keys: Vec<SdfKey>,
}

impl AppState {
    pub fn new(sdf_iterator: SdfScene) -> AppState {
        let cam = FirstPersonCamera::new(sdf_iterator.eye_pos);
        let shader = create_shader_program();
        AppState {
            sdf_iterator,
            nodes: HashMap::new(),
            camera: cam,
            sdf_cache: HashSet::new(),
            shader,
            time: 0.0,
            wireframe: false,
            voxels: false,
            visible_keys: Vec::new(),
        }
    }

    pub fn clear_cache(&mut self) {
        println!("Reloading cache!");
        for (_, (mesh, _, _, _)) in self.nodes.iter() {
            mesh.destroy();
        }
        self.nodes.clear();
    }

    pub fn toggle_wireframe(&mut self) {
        self.wireframe = !self.wireframe;
        self.clear_cache();
    }

    pub fn toggle_voxels(&mut self) {
        self.voxels = !self.voxels;
        self.clear_cache();
    }

    pub fn step(&mut self, win_width: u32, win_height: u32) {
        self.time += 1.0;

        let centerpos = self.camera.position.map(|x| f32::floor(x / 16.0) * 16.0);
        self.sdf_iterator.eye_pos = self.camera.position;

        let aspect = win_width as f32 / win_height.max(1) as f32;
        let proj = mat4_perspective(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0);
        let view = self.camera.view_matrix();
        let vp = mat4_mul(&proj, &view);
        self.sdf_iterator.cam = vp;

        self.sdf_iterator.iterate_scene(centerpos, 8.0 * 128.0 * 2.0);

        // Track which keys are visible this frame
        self.visible_keys.clear();

        let mut reload_count = 0;
        for block in &self.sdf_iterator.render_blocks.clone() {
            loop {
                let needs_insert = !self.nodes.contains_key(&block.2)
                    || self.nodes.get(&block.2).map_or(false, |nd| !nd.1.eq(&block.3));

                if needs_insert {
                    // Remove old if it exists
                    if let Some(old) = self.nodes.remove(&block.2) {
                        reload_count += 1;
                        println!(
                            "Reload: {} ({} {} {} {})",
                            reload_count, block.2.x, block.2.y, block.2.z, block.2.w
                        );
                        old.0.destroy();
                    }

                    let sdf2 = block.3.cached(&mut self.sdf_cache);
                    let pos = block.2;
                    let size = pos.w as f32;

                    let mut r = TriangleList::new();
                    let newsdf = sdf2
                        .optimized_for_block(block.0.into(), size, &mut self.sdf_cache)
                        .cached(&mut self.sdf_cache)
                        .clone();

                    if self.voxels {
                        marching_voxels(
                            &mut r,
                            &newsdf,
                            block.0.into(),
                            size,
                            0.2 * 2.0_f32.powf(block.4),
                            size,
                            &mut self.sdf_cache,
                        );
                    } else {
                        marching_cubes_sdf(
                            &mut r,
                            &newsdf,
                            block.0.into(),
                            size,
                            0.4 * 2.0_f32.powf(block.4),
                            size,
                        );
                    }

                    if r.any() {
                        let raw_mesh = r.to_mesh(&newsdf);
                        let gl_mesh = GlMesh::from_raw(&raw_mesh);
                        self.nodes.insert(block.2, (gl_mesh, block.3.clone(), size, block.0));
                    } else {
                        // Empty mesh - create a placeholder with 0 indices
                        let empty = GlMesh {
                            vbo_pos: 0,
                            vbo_uv: 0,
                            ebo: 0,
                            texture: 0,
                            num_indices: 0,
                        };
                        self.nodes.insert(block.2, (empty, block.3.clone(), size, block.0));
                    }
                }
                self.visible_keys.push(block.2);
                break;
            }
        }

        // Render
        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::DEPTH_TEST);

            if self.wireframe {
                gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
            } else {
                gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
            }

            gl::UseProgram(self.shader.id);
            gl::UniformMatrix4fv(self.shader.loc_mvp, 1, gl::FALSE, vp.as_ptr());
            gl::Uniform1i(self.shader.loc_texture, 0);

            for key in &self.visible_keys {
                if let Some((mesh, _, _, _)) = self.nodes.get(key) {
                    if mesh.num_indices == 0 {
                        continue;
                    }

                    gl::ActiveTexture(gl::TEXTURE0);
                    gl::BindTexture(gl::TEXTURE_2D, mesh.texture);

                    gl::EnableVertexAttribArray(self.shader.loc_pos);
                    gl::BindBuffer(gl::ARRAY_BUFFER, mesh.vbo_pos);
                    gl::VertexAttribPointer(
                        self.shader.loc_pos,
                        3,
                        gl::FLOAT,
                        gl::FALSE,
                        0,
                        ptr::null(),
                    );

                    gl::EnableVertexAttribArray(self.shader.loc_uv);
                    gl::BindBuffer(gl::ARRAY_BUFFER, mesh.vbo_uv);
                    gl::VertexAttribPointer(
                        self.shader.loc_uv,
                        2,
                        gl::FLOAT,
                        gl::FALSE,
                        0,
                        ptr::null(),
                    );

                    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, mesh.ebo);
                    gl::DrawElements(
                        gl::TRIANGLES,
                        mesh.num_indices,
                        gl::UNSIGNED_SHORT,
                        ptr::null(),
                    );

                    gl::DisableVertexAttribArray(self.shader.loc_pos);
                    gl::DisableVertexAttribArray(self.shader.loc_uv);
                }
            }
        }
    }

    pub fn handle_right_click(&mut self, ray_origin: Vec3, ray_dir: Vec3) {
        let col = self.sdf_iterator.sdf.cast_ray(ray_origin, ray_dir, 1000.0);
        if let Some((_, p)) = col {
            let newobj = Sphere::new(p, 5.0);
            let sub = self.sdf_iterator.sdf.clone().subtract(newobj.into());
            self.sdf_iterator.sdf = sub.into();
            println!("old: {}", self.sdf_iterator.sdf);
            self.sdf_iterator.sdf = self.sdf_iterator.sdf.optimize_bounds();
            println!("new: {}", self.sdf_iterator.sdf);
        }
    }
}

impl Drop for AppState {
    fn drop(&mut self) {
        for (_, (mesh, _, _, _)) in self.nodes.iter() {
            mesh.destroy();
        }
        unsafe {
            gl::DeleteProgram(self.shader.id);
        }
    }
}
