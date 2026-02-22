extern crate gl;
extern crate glfw;
mod octree2;

use supersdf::color::*;
use supersdf::sdf::*;
use supersdf::sdf_compiler;
use supersdf::vec3::Vec3;
use supersdf::mat4::{self, Frustum};
use octree2::{build_octree, OctreeNode as OctreeNode2};

use gl::types::*;
use glfw::{Action, Context, Key, MouseButton};
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::rc::Rc;
use std::str;
use std::time::Instant;
use rand::{Rng, SeedableRng, rngs::StdRng};

const VOXEL_VERTEX_SHADER_SRC: &str = r#"
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 iChunkPos;
layout (location = 2) in uint iAtlasLayer;
layout (location = 3) in float iChunkSize;

out vec3 vLocalPos;
out vec3 vWorldPos;
flat out vec3 vChunkOrigin;
flat out uint vAtlasLayer;

uniform mat4 uViewProj;

void main()
{
    vLocalPos    = aPos;
    vAtlasLayer  = iAtlasLayer;
    vChunkOrigin = iChunkPos;

    vec3 worldPos = iChunkPos + aPos * iChunkSize;
    vWorldPos     = worldPos;
    gl_Position   = uViewProj * vec4(worldPos, 1.0);
}
"#;
const VOXEL_FRAGMENT_SHADER_SRC: &str = r#"
#version 330 core

in  vec3 vLocalPos;      // [0,1]^3 on the cube face
in  vec3 vWorldPos;      // world-space position of this fragment
flat in vec3 vChunkOrigin; // world-space min corner of this chunk
flat in uint vAtlasLayer;

// GL_TEXTURE_2D_ARRAY: 16×4×N. Layout: col = x + z*4, row = y, layer = brick index.
uniform sampler2DArray uBrickAtlas;
uniform sampler1D      uPalette;     // 256-entry RGB palette
uniform vec3           uCameraPos;

out vec4 FragColor;

float sample_voxel(ivec3 p)
{
    return texelFetch(
        uBrickAtlas,
        ivec3(p.x + p.z * 4, p.y, int(vAtlasLayer)),
        0
    ).r;
}

void main()
  {
      vec3 rayDir = normalize(vWorldPos - uCameraPos);
      vec3 pos = clamp(vLocalPos * 4.0, vec3(0.001), vec3(3.999));

      // DDA setup
      ivec3 voxel = ivec3(floor(pos));
      ivec3 stepDir  = ivec3(sign(rayDir));
      vec3 tDelta = abs(1.0 / rayDir);
      vec3 fpos = fract(pos);
      vec3 tMax = vec3(
          tDelta.x * (rayDir.x >= 0.0 ? (1.0 - fpos.x) : fpos.x),
          tDelta.y * (rayDir.y >= 0.0 ? (1.0 - fpos.y) : fpos.y),
          tDelta.z * (rayDir.z >= 0.0 ? (1.0 - fpos.z) : fpos.z)
      );

      for (int i = 0; i < 12; i++) {
          if (any(lessThan(voxel, ivec3(0))) || any(greaterThan(voxel, ivec3(3))))
              break;

          float v = sample_voxel(voxel);
          if (v > 0.001) {
              // v is palette index / 255.0; recover index and look up color
              int palIdx = int(v * 255.0 + 0.5);
              vec3 color = texelFetch(uPalette, palIdx, 0).rgb;
              FragColor = vec4(color, 1.0);
              return;
          }

          // Advance along the axis with smallest tMax
          if (tMax.x < tMax.y && tMax.x < tMax.z) {
              voxel.x += stepDir.x;  tMax.x += tDelta.x;
          } else if (tMax.y < tMax.z) {
              voxel.y += stepDir.y;  tMax.y += tDelta.y;
          } else {
              voxel.z += stepDir.z;  tMax.z += tDelta.z;
          }
      }

      discard;
  }"#;

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

fn calculate_lod(distance: f32, max_distance: f32, num_lods: u32) -> u32 {
    let ratio = (distance / max_distance).clamp(0.0, 1.0);
    let lod = (ratio * num_lods as f32) as u32;
    lod.min(num_lods - 1)
}


// ---------- Scene ----------

fn build_initial_scene() -> DistanceFieldEnum {
    let mut sdf: DistanceFieldEnum = DistanceFieldEnum::Empty;
    let mut rng = StdRng::seed_from_u64(42);
    
    let field_size = 4000;

    for i in (-field_size..field_size).step_by(10) {
        for j in (-field_size..field_size).step_by(10) {
            let x = i as f32 + rng.gen_range(-2.0..2.0);
            let z = j as f32 + rng.gen_range(-2.0..2.0);
            let y = -20.0 + rng.gen_range(-2.0..2.0);
            let r = rng.gen_range(8.0..10.0);
            let color = Color::rgb(
                (rng.gen_range(0.1..4.0) as f32).floor() * 0.25,
                (rng.gen_range(0.1..4.0) as f32).floor() * 0.25,
                (rng.gen_range(0.1..4.0) as f32).floor() * 0.25,
            );
            sdf = sdf.insert_2(DistanceFieldEnum::sphere(Vec3::new(x, y, z), r).colored(color));
        }
    }
    sdf = sdf.insert_2(DistanceFieldEnum::aabb(Vec3::new(0.0, -2018.0, 0.0), Vec3::new(field_size as f32, 2000.0, field_size as f32)));
    sdf.optimize_bounds()
}

#[repr(C)]
#[derive(Clone)]
pub struct VoxelChunk {
    // 4x4x4 = 64 voxels
    pub voxels: [u8; 64],
}

#[repr(C)]
pub struct VoxelInstanceData {
    pub chunk_pos: [f32; 3],   // world position
    pub atlas_layer: u32,      // brick index (layer within its super chunk's texture)
    pub chunk_size: f32,       // 4.0 for LOD 0, 8.0 for LOD 1, etc.
}

const MAX_LAYERS_PER_TEXTURE: usize = 2048;
const LOD_FACTOR: f32 = 60.0;   // use coarse LOD when distance > size * LOD_FACTOR
const MAX_LOD_SIZE: f32 = 64.0; // max branch size to voxelize for LOD

/// Palette: maps u8 index (1-255) to RGB color. Index 0 = air/empty.
struct Palette {
    colors: Vec<[u8; 3]>,              // palette entries (index 0 = unused placeholder)
    lookup: HashMap<[u8; 3], u8>,       // RGB -> palette index for dedup
}

impl Palette {
    fn new() -> Self {
        Palette {
            colors: vec![[0, 0, 0]], // index 0 = air
            lookup: HashMap::new(),
        }
    }

    /// Get or insert a color, returning its palette index (1-255). Returns 1 if full.
    fn get_or_insert(&mut self, color: Color) -> u8 {
        let rgb = [
            (color.r.clamp(0.0, 1.0) * 255.0) as u8,
            (color.g.clamp(0.0, 1.0) * 255.0) as u8,
            (color.b.clamp(0.0, 1.0) * 255.0) as u8,
        ];
        if let Some(&idx) = self.lookup.get(&rgb) {
            return idx;
        }
        if self.colors.len() >= 256 {
            return 1; // palette full, use first color
        }
        let idx = self.colors.len() as u8;
        self.colors.push(rgb);
        self.lookup.insert(rgb, idx);
        idx
    }
}

/// Voxelize an octree node (leaf or branch) at 4x4x4 resolution.
/// Voxel values are palette indices (0=air, 1-255=color).
fn voxelize_node(center: &Vec3, size: f32, sdf: &DistanceFieldEnum, palette: &mut Palette) -> Option<VoxelChunk> {
    let step = size / 4.0;
    let half = size / 2.0;
    let mut chunk = VoxelChunk { voxels: [0; 64] };
    let mut any = false;
    let mut index = 0;
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let pt = Vec3::new(
                    center.x - half + (x as f32 + 0.5) * step,
                    center.y - half + (y as f32 + 0.5) * step,
                    center.z - half + (z as f32 + 0.5) * step,
                );
                let d = sdf.distance(pt);
                if d < step * 0.5 && d > -step {
                    let color = sdf.color(pt);
                    chunk.voxels[index] = palette.get_or_insert(color);
                    any = true;
                }
                index += 1;
            }
        }
    }
    if any { Some(chunk) } else { None }
}

struct SuperChunk {
    tex: GLuint,           // GL_TEXTURE_2D_ARRAY, up to MAX_LAYERS_PER_TEXTURE layers
    vao: GLuint,           // VAO with cube geometry + per-instance data
    instance_vbo: GLuint,  // VBO for VoxelInstanceData
    instances: u32
    
}

unsafe fn upload_chunk(tex: u32, layer: u32, chunk: &VoxelChunk) {
    // Repack 4×4×4 (z-major) into 16×4 (row=y, col=x+z*4) for a 2D array layer.
    // Source layout: voxels[z*16 + y*4 + x]
    let mut buf = [0u8; 64];
    for z in 0..4usize {
        for y in 0..4usize {
            for x in 0..4usize {
                buf[y * 16 + (x + z * 4)] = chunk.voxels[z * 16 + y * 4 + x];
            }
        }
    }

    gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex);
    gl::TexSubImage3D(
        gl::TEXTURE_2D_ARRAY,
        0,
        0, 0, layer as i32, // x_off, y_off, layer index
        16, 4, 1,           // width=16, height=4, one layer
        gl::RED,
        gl::UNSIGNED_BYTE,
        buf.as_ptr() as *const _,
    );
}

fn get_superchunk(node: &octree2::OctreeNode, center: Vec3, size: f32, palette: &mut Palette, cube_vbo: GLuint, min_size: f32) -> SuperChunk {

    let mut tex: GLuint = 0;
    unsafe {
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex);
        gl::TexStorage3D(gl::TEXTURE_2D_ARRAY, 1, gl::R8, 16, 4, 1024);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_BASE_LEVEL, 0);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAX_LEVEL, 0);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
    }

    let mut layer = vec![];
    
    {
        
        let mut stack: Vec<&octree2::OctreeNode> = vec![node];
        while let Some(n) = stack.pop() {
            let mut chunk = None;
            match n {
                octree2::OctreeNode::Empty => {}
                octree2::OctreeNode::Leaf { center, size, sdf } => {
                    // reached 4x4x4 block
                    chunk = voxelize_node(center, *size, sdf, palette);
                    let half = *size / 2.0;
                    layer.push(VoxelInstanceData{
                        chunk_pos: [center.x - half, center.y - half, center.z - half],
                        atlas_layer: layer.len() as u32,
                        chunk_size: *size 
                    });
                }
                octree2::OctreeNode::Branch { center, size, sdf, children } => {
                    if *size <= min_size {
                        
                        chunk = voxelize_node(center, *size, sdf, palette);
                        let half = *size / 2.0;
                        layer.push(VoxelInstanceData{
                            chunk_pos: [center.x - half, center.y - half, center.z - half],
                            atlas_layer: layer.len() as u32,
                            chunk_size: *size
                        });
                        
                    }else{
                        for child in children.iter().flatten() {
                            stack.push(child.as_ref());
                        }
                    }
                
                }
            }
            if let Some (chunk2) = chunk {
                unsafe {
                    upload_chunk(tex, layer.len() as u32 - 1, &chunk2);
                }
                if layer.len() > 1024 * 2 {
                    panic!("Too many layers!");
                }
            }
        }
    }
    
    let mut sc_vao: GLuint = 0;
    let mut sc_instance_vbo: GLuint = 0;
    unsafe {
        gl::GenVertexArrays(1, &mut sc_vao);
        gl::BindVertexArray(sc_vao);
        
        gl::BindBuffer(gl::ARRAY_BUFFER, cube_vbo);
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE,
                                (3 * std::mem::size_of::<f32>()) as GLsizei, ptr::null());
        gl::EnableVertexAttribArray(0);
        
        gl::GenBuffers(1, &mut sc_instance_vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, sc_instance_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (layer.len() * std::mem::size_of::<VoxelInstanceData>()) as GLsizeiptr,
            layer.as_ptr() as *const _,
            gl::DYNAMIC_DRAW,
            );
        println!("Layer len: {}", layer.len());
        let stride = std::mem::size_of::<VoxelInstanceData>() as GLsizei;
        
        gl::VertexAttribPointer(1, 3, gl::FLOAT, gl::FALSE, stride, ptr::null());
        gl::EnableVertexAttribArray(1);
        gl::VertexAttribDivisor(1, 1);
        
        gl::VertexAttribIPointer(2, 1, gl::UNSIGNED_INT, stride,
                                 (3 * std::mem::size_of::<f32>()) as *const _);
        gl::EnableVertexAttribArray(2);
        gl::VertexAttribDivisor(2, 1);
        
        gl::VertexAttribPointer(3, 1, gl::FLOAT, gl::FALSE, stride,
                                (4 * std::mem::size_of::<f32>()) as *const _);
        gl::EnableVertexAttribArray(3);
        gl::VertexAttribDivisor(3, 1);
        
        gl::BindVertexArray(0);
    }
    SuperChunk {
        tex: tex,
        vao: sc_vao,
        instance_vbo: sc_instance_vbo,
        instances: layer.len() as u32
    }
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

    let mut palette = Palette::new();
    let mut palette_tex: GLuint = 0;
    unsafe {
        gl::GenTextures(1, &mut palette_tex);
    }        
    let mut octree2 = build_octree(&sdf, 2048.0);
    let mut palette_colors =0;
    
    let mut node_instance_lookup = HashMap::new();

    let voxel_prog = {
        let vs = compile_gl_shader(VOXEL_VERTEX_SHADER_SRC, gl::VERTEX_SHADER);
        let fs = compile_gl_shader(VOXEL_FRAGMENT_SHADER_SRC, gl::FRAGMENT_SHADER);
        let id = link_program(vs, fs);
        (
            id,
            unsafe { gl::GetUniformLocation(id, CString::new("uViewProj").unwrap().as_ptr()) },
            unsafe { gl::GetUniformLocation(id, CString::new("uBrickAtlas").unwrap().as_ptr()) },
            unsafe { gl::GetUniformLocation(id, CString::new("uCameraPos").unwrap().as_ptr()) },
            unsafe { gl::GetUniformLocation(id, CString::new("uPalette").unwrap().as_ptr()) },
        )
    };

    // Camera state
    let mut cam_pos = Vec3::new(0.0, 0.0, -60.0);
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.0;
    let mut last_cursor = (0.0f64, 0.0f64);
    let mut first_mouse = true;
    let move_speed = 1.0f32;
    let mouse_sensitivity = 0.002f32;
    let mut left_mouse_down = false;

    let mut key_w = false;
    let mut key_a = false;
    let mut key_s = false;
    let mut key_d = false;

    // FPS counter
    let mut fps_last_time = Instant::now();
    let mut fps_frame_count: u32 = 0;


    
    // Toggle this to switch between SDF path-tracing and voxel rendering
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
                        Key::B if pressed => {
                            
                        }
                        _ => {}
                    }
                }
                glfw::WindowEvent::MouseButton(MouseButton::Button1, action, _) => {
                    left_mouse_down = action == Action::Press;
                    if action == Action::Press {
                        first_mouse = true;
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

                    if let Some((_dist, hit_pos)) = sdf.cast_ray(cam_pos, ray_dir, 10000.0) {
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
                    if left_mouse_down {
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
                    } else {
                        last_cursor = (x, y);
                    }
                }
                _ => {}
            }
        }

        // Rebuild voxel map if SDF changed
        if sdf_dirty {
            sdf_dirty = false;
            let mut cache = HashSet::new();
            let mut reused_count = 0u32;
            octree2 = OctreeNode2::build_node(Vec3::ZERO, 2048.0 * 4.0, &sdf, &octree2, &mut cache, &mut reused_count);
        
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
        let view = mat4::view(cam_pos, dir, up);
        let proj = mat4::perspective(fovy, aspect, 0.1, 4000.0);
        let vp = mat4::mul(&proj, &view);

        unsafe {
            // Sky background
            gl::ClearColor(0.25, 0.35, 0.55, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);

            gl::BindVertexArray(vao);

                let frustum = Frustum::from_vp(&vp);
                let mut to_render = vec![];
                {
                    let mut oct_stack: Vec<&octree2::OctreeNode> = vec![&octree2];
                    while let Some(node) = oct_stack.pop() {
                    match node {
                        octree2::OctreeNode::Empty => {}
                        octree2::OctreeNode::Leaf { center, size, .. } => {
                            
                            if frustum.cull_aabb(*center, *size / 2.0) { continue; }
                            panic!("this should never happen!");
                        }
                        octree2::OctreeNode::Branch { center, size, children, .. } => {
                            if frustum.cull_aabb(*center, *size / 2.0) { continue; }
                            
                            let dist = (*center - cam_pos).length();
                            
                            let lod = calculate_lod(dist, 2000.0, 5) + 1;
                            
                            if *size <= 64.0 * (lod as f32) {
                                if node_instance_lookup.contains_key(node) == false {
                                    let chunk = get_superchunk(node, *center, *size, &mut palette, vbo, (lod * 4) as f32);                           
                                    node_instance_lookup.insert(node.clone(), chunk);
                                    
                                }
                                to_render.push(node);
         
                                continue;
                            }
                            

                            // Recurse into children front-to-back
                            let near = ((cam_pos.x > center.x) as usize)
                                | (((cam_pos.y > center.y) as usize) << 1)
                                | (((cam_pos.z > center.z) as usize) << 2);
                            for &mask in &[7usize, 6, 5, 3, 4, 2, 1, 0] {
                                if let Some(ref child) = children[near ^ mask] {
                                    oct_stack.push(child.as_ref());
                                }
                            }
                        }
                    }
                }
            }

                
            // Voxel rendering with LOD: per-frame octree traversal
            if !to_render.is_empty() {

                if palette_colors != palette.colors.len() {
                    println!("update palette!");
                    palette_colors = palette.colors.len();
                    unsafe {
                    
                        gl::BindTexture(gl::TEXTURE_1D, palette_tex);
                        // Pad palette to 256 entries
                        let mut palette_data = [[0u8; 3]; 256];
                        for (i, c) in palette.colors.iter().enumerate() {
                            palette_data[i] = *c;
                        }
                        gl::TexImage1D(
                            gl::TEXTURE_1D, 0, gl::RGB8 as i32,
                            256, 0, gl::RGB, gl::UNSIGNED_BYTE,
                            palette_data.as_ptr() as *const _,
                            );
                        gl::TexParameteri(gl::TEXTURE_1D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
                        gl::TexParameteri(gl::TEXTURE_1D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
                    }
                }

                
                // Upload instance data and draw
                gl::UseProgram(voxel_prog.0);
                gl::UniformMatrix4fv(voxel_prog.1, 1, gl::FALSE, vp.as_ptr());
                gl::Uniform1i(voxel_prog.2, 0);  // brick atlas on texture unit 0
                gl::Uniform1i(voxel_prog.4, 1);  // palette on texture unit 1
                gl::Uniform3f(voxel_prog.3, cam_pos.x, cam_pos.y, cam_pos.z);

                // Bind palette texture to unit 1
                gl::ActiveTexture(gl::TEXTURE1);
                gl::BindTexture(gl::TEXTURE_1D, palette_tex);
                gl::ActiveTexture(gl::TEXTURE0);

                for (i, node) in to_render.iter().enumerate() {
                    if let Some(sc) = node_instance_lookup.get(&node) {
                    
                        gl::BindBuffer(gl::ARRAY_BUFFER, sc.instance_vbo);
                        gl::BindTexture(gl::TEXTURE_2D_ARRAY, sc.tex);
                        gl::BindVertexArray(sc.vao);
                        gl::DrawArraysInstanced(gl::TRIANGLES, 0, 36, sc.instances as GLsizei);
                    }
                }
                gl::BindVertexArray(0);
            }
        }

        window.swap_buffers();

        // FPS counter
        fps_frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(fps_last_time).as_secs_f64();
        if elapsed >= 1.0 {
            println!("FPS: {:.1}", fps_frame_count as f64 / elapsed);
            fps_frame_count = 0;
            fps_last_time = now;
        }
    }
    unsafe {
        gl::DeleteProgram(voxel_prog.0);
        gl::DeleteVertexArrays(1, &vao);
        gl::DeleteBuffers(1, &vbo);
    }
}
