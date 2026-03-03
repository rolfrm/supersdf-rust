extern crate gl;
extern crate glfw;
mod octree2;

use supersdf::color::*;
use supersdf::sdf::*;
use supersdf::vec3::Vec3;
use supersdf::mat4::{self, Frustum};
use octree2::{build_octree, fast_cast_ray, OctreeNode as OctreeNode2};

use gl::types::*;
use glfw::{Action, Context, Key, MouseButton};
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;
use std::rc::Rc;
use std::str;
use std::time::Instant;
use rand::{Rng, SeedableRng, rngs::StdRng};

const CURSOR_VERTEX_SHADER_SRC: &str = r#"
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uViewProj;
uniform vec3 uCenter;
uniform float uSize;
void main() {
    gl_Position = uViewProj * vec4(uCenter + aPos * uSize, 1.0);
}
"#;

const CURSOR_FRAGMENT_SHADER_SRC: &str = r#"
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"#;

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

in  vec3 vLocalPos;
in  vec3 vWorldPos;
flat in vec3 vChunkOrigin;
flat in uint vAtlasLayer;

// GL_TEXTURE_2D_ARRAY: 16×4×N. Layout: col = x + z*4, row = y, layer = brick index.
uniform sampler2DArray uBrickAtlas;
uniform sampler2DArray uNormalAtlas;
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

vec4 sample_normal_ao(ivec3 p)
{
    vec4 raw = texelFetch(
        uNormalAtlas,
        ivec3(p.x + p.z * 4, p.y, int(vAtlasLayer)),
        0
    );
    vec3 n = normalize(raw.rgb * 2.0 - 1.0);
    return vec4(n, raw.a); // xyz = normal, w = ao
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
          if (v > 0.000001) {
              // v is palette index / 255.0; recover index and look up color
              int palIdx = int(v * 65535.0 + 0.5);
              if(palIdx == 1) discard;
              vec3 color = texelFetch(uPalette, palIdx, 0).rgb;

              // Decode normal + AO (separate channels)
              vec4 nao = sample_normal_ao(voxel);
              vec3 normal = nao.xyz;
              float ao = nao.w;
              vec3 lightDir = normalize(vec3(0.4, 0.8, 0.3));
              float ambient = 0.25;
              float diffuse = max(dot(normal, lightDir), 0.0);
              color *= (ambient + diffuse * 0.75) * ao;

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


fn build_wood() -> Coloring {
    Noise::new(91823, Color::rgb(0.55, 0.33, 0.14), Color::rgb(0.36, 0.2, 0.09))
        .scaled( Vec3::new(3.0, 0.5 * 0.2, 3.0))
        .mix_with(0.5, Noise::new(4732132,  Color::rgb(0.55, 0.33, 0.14), Color::rgb(0.36, 0.2, 0.09)))
        
}

fn build_grass() -> Coloring {
    Noise::new(38291, Color::rgb(0.2, 0.5, 0.1), Color::rgb(0.15, 0.35, 0.05))
        .scaled(Vec3::new(2.0, 0.3, 2.0))
        .mix_with(0.5, Noise::new(7291044, Color::rgb(0.25, 0.55, 0.12), Color::rgb(0.1, 0.3, 0.04)))
}

fn build_stone() -> Coloring {
    Noise::new(55012, Color::rgb(0.45, 0.44, 0.42), Color::rgb(0.32, 0.31, 0.30))
        .scaled(Vec3::new(1.5, 1.5, 1.5))
        .mix_with(0.5, Noise::new(1198374, Color::rgb(0.5, 0.48, 0.45), Color::rgb(0.28, 0.27, 0.26)))
}

fn build_box(center: Vec3, size: Vec3, wall_thickness: f32) -> Sdf {
    let t = Vec3::new(wall_thickness, wall_thickness, wall_thickness);
    Aabb::new(center, size).subtract(Aabb::new(center, size - t))
}
// ---------- Scene ----------

fn build_initial_scene0() -> Sdf {
    let mut rng = StdRng::seed_from_u64(42);

    let field_size = FIELD_SIZE;
    let mut items = vec![];
    
    let poles = 2;

    {
        let pole = Sdf::aabb(Vec3::new(0.0, -1010.0, 0.0), Vec3::new(10000.0, 1000.0, 10000.0));
        let wood = pole.with_coloring(build_grass());
        items.push(Rc::new(wood));
    }

    for i in -10..10{
        let pole = Sdf::sphere(Vec3::new(100.0, -10.0, i as f32 * 100.0), 50.0);
        let wood = pole.with_coloring(build_stone());
        items.push(Rc::new(wood));
    }
    
    for i in -poles..poles {
        for j in -poles..poles {
            let pole = Sdf::aabb(Vec3::new(i as f32 * 5.0, 0.0, j as f32 * 5.0), Vec3::new(1.0, 5.0, 1.0));
            // Wood texture: coarse grain noise stretched vertically, with fine detail noise on top
            let wood = pole.with_coloring(build_wood());
            items.push(Rc::new(wood));
        }
    }


    items.push(Rc::new(Sdf::aabb(Vec3::new(-40.0, 0.0, 0.0), Vec3::new(0.5, 10.0, 41.0))
        .with_coloring(ColorScale::new(Vec3::new(1.0, 0.5, 1.0), Noise::new(132132147, Color::rgb(0.5, 0.3, 0.2), Color::rgb(0.3, 0.5, 0.7))))));
    items.push(Rc::new(Sdf::aabb(Vec3::new(40.0, 0.0, 0.0), Vec3::new(0.5, 10.0, 41.0)).with_color(Color::rgb(0.5, 0.4, 0.3))));
    items.push(Rc::new(Sdf::aabb(Vec3::new(0.0, 0.0, 40.0), Vec3::new(40.0, 10.0, 0.25)).with_color(Color::rgb(0.5, 0.4, 0.3))));
    items.push(Rc::new(Sdf::aabb(Vec3::new(00.0, 0.0, -40.0), Vec3::new(40.0, 10.0, 0.25)).with_color(Color::rgb(0.5, 0.4, 0.3))));

    items.push(Rc::new(build_box(Vec3::new(-100.0,10.0, 0.0), Vec3::new(50.0, 20.0, 50.0), 2.0)
                       .with_coloring(build_wood())));
    

    let sdf: Sdf = Add::from_items(items).into();//_subdivide(items, 4).into();
    //let sdf2 = sdf.optimized_for_block(Vec3::ZERO, 1000.0);
    let sdf2 = sdf.optimized_for_block(Vec3::ZERO, (field_size as f32) * 4.0);

    return (*sdf2).clone();
        
}


fn build_initial_scene() -> Sdf {
    let mut rng = StdRng::seed_from_u64(42);

    let field_size = FIELD_SIZE;
    let mut items = vec![];
    for i in (-field_size..field_size).step_by(10) {
        for j in (-field_size..field_size).step_by(10) {
            let x = i as f32 + rng.gen_range(-2.0..2.0);
            let z = j as f32 + rng.gen_range(-2.0..2.0);
            let y = -20.0 + rng.gen_range(-2.0..2.0);
            let r = rng.gen_range(8.0..10.0);
            let color = Color::rgb(
                (rng.gen_range(0.1..3.0) as f32).floor() * 0.33,
                (rng.gen_range(0.1..3.0) as f32).floor() * 0.33,
                (rng.gen_range(0.1..3.0) as f32).floor() * 0.33,
            );
            items.push(Rc::new(Sdf::sphere(Vec3::new(x, y, z), r).with_color(color)));
        }
    }

    let poles = 2;

    for i in -poles..poles {
        for j in -poles..poles {
            let pole = Sdf::aabb(Vec3::new(i as f32 * 5.0, 0.0, j as f32 * 5.0), Vec3::new(1.0, 5.0, 1.0));
            // Wood texture: coarse grain noise stretched vertically, with fine detail noise on top
            let wood = pole.with_coloring(build_wood());
            items.push(Rc::new(wood));
        }
    }


    items.push(Rc::new(Sdf::aabb(Vec3::new(-40.0, 0.0, 0.0), Vec3::new(0.5, 10.0, 41.0))
        .with_coloring(ColorScale::new(Vec3::new(1.0, 0.5, 1.0), Noise::new(132132147, Color::rgb(0.5, 0.3, 0.2), Color::rgb(0.3, 0.5, 0.7))))));
    items.push(Rc::new(Sdf::aabb(Vec3::new(40.0, 0.0, 0.0), Vec3::new(0.5, 10.0, 41.0)).with_color(Color::rgb(0.5, 0.4, 0.3))));
    items.push(Rc::new(Sdf::aabb(Vec3::new(0.0, 0.0, 40.0), Vec3::new(40.0, 10.0, 0.25)).with_color(Color::rgb(0.5, 0.4, 0.3))));
    items.push(Rc::new(Sdf::aabb(Vec3::new(00.0, 0.0, -40.0), Vec3::new(40.0, 10.0, 0.25)).with_color(Color::rgb(0.5, 0.4, 0.3))));

    items.push(Rc::new(build_box(Vec3::new(-100.0,40.0, 0.0), Vec3::new(50.0, 50.0, 50.0), 2.0)
                       .with_coloring(build_wood())));
    

    let sdf: Sdf = Add::from_items(items).into();//_subdivide(items, 4).into();
    //let sdf2 = sdf.optimized_for_block(Vec3::ZERO, 1000.0);
    let sdf2 = sdf.optimized_for_block(Vec3::ZERO, (field_size as f32) * 4.0);

    return (*sdf2).clone();
        
}

#[repr(C)]
#[derive(Clone)]
pub struct VoxelChunk {
    // 4x4x4 = 64 voxels
    pub voxels: [u16; 64],
    // Packed normals: 4 bytes (nx, ny, nz, ao) per voxel; xyz mapped from [-1,1] to [0,255], w = ao [0,255]
    pub normals: [[u8; 4]; 64],
}

#[repr(C)]
pub struct VoxelInstanceData {
    pub chunk_pos: [f32; 3],   // world position
    pub atlas_layer: u32,      // brick index (layer within its super chunk's texture)
    pub chunk_size: f32,       // 4.0 for LOD 0, 8.0 for LOD 1, etc.
}

const ROOT_SIZE : f32= 32000.0;
const FIELD_SIZE: i32 = 5000;

/// Palette: maps u8 index (1-255) to RGB color. Index 0 = air/empty.
struct Palette {
    colors: Vec<[u8; 3]>,              // palette entries (index 0 = unused placeholder)
    lookup: HashMap<[u8; 3], u16>,       // RGB -> palette index for dedup
}

impl Palette {
    fn new() -> Self {
        Palette {
            colors: vec![[0, 0, 0],
                         [0,0,0]], // index 0 = air, // index 1 = not visible
            lookup: HashMap::new(),
        }
    }

    /// Get or insert a color, returning its palette index (1-255). Returns 1 if full.
    fn get_or_insert(&mut self, color: Color) -> u16 {
        let rgb = [
            (color.r.clamp(0.0, 1.0) * 255.0) as u8,
            (color.g.clamp(0.0, 1.0) * 255.0) as u8,
            (color.b.clamp(0.0, 1.0) * 255.0) as u8,
        ];
        if let Some(&idx) = self.lookup.get(&rgb) {
            return idx;
        }
        if self.colors.len() >= 256 * 256 {
            return 128; // palette full, use first color
        }
        let idx = self.colors.len() as u16;
        self.colors.push(rgb);
        self.lookup.insert(rgb, idx);
        idx
    }
}

/// Voxelize an octree node (leaf or branch) at 4x4x4 resolution.
/// Voxel values are palette indices (0=air, 1-255=color).
fn voxelize_node(center: Vec3, size: f32, sdf: &Sdf, palette: &mut Palette) -> Option<VoxelChunk> {
    let step = size / 4.0;
    let half = size / 2.0;
    let eps = step * 0.25;
    let mut chunk = VoxelChunk { voxels: [0; 64], normals: [[128, 128, 128, 255]; 64] };
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
                if d < -step * 0.8 {
                    chunk.voxels[index] = 1;
                }
                else if d < step * 0.8 {
                    let color = sdf.color(pt);
                    chunk.voxels[index] = palette.get_or_insert(color);
                    let grad = sdf.gradient_at(pt, eps);
                    let len = grad.length();
                    let n = if len > 1e-6 { grad * (1.0 / len) } else { Vec3::new(0.0, 1.0, 0.0) };

                    let grad2 = sdf.gradient_at(pt + n * size, eps);
                    let len2 = grad2.length();
                    let n2 = if len2 > 1e-6 {grad2 * (1.0 / len2) } else {Vec3::new(0.0, 1.0, 0.0)}; 
                    let pt2 = pt + n2;
                    // AO: step along normal at exponential distances, compare
                    // actual SDF distance to expected (unoccluded) distance
                    let mut ao = 0.0f32;
                    let mut t = size;
                    let num_steps = 0u32;
                    for _ in 0..num_steps {
                        let sample_pt = pt2 + n2 * t;
                        let sample_d = sdf.distance(sample_pt);
                        ao += (sample_d / t).clamp(0.0, 1.0);
                        t *= 2.0;
                    }

                    if num_steps == 0 {
                        ao = 1.0;
                    }else{
                        ao /= num_steps as f32;
                    }
                    ao = ao.clamp(0.0, 1.0);
                    
                    // Encode normal (xyz) and AO (w) separately for full precision
                    chunk.normals[index] = [
                        ((n.x * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8,
                        ((n.y * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8,
                        ((n.z * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8,
                        (ao * 255.0).clamp(0.0, 255.0) as u8,
                    ];
                    any = true;
                }
                index += 1;
            }
        }
    }
    if any { Some(chunk) } else { None }
}

struct SuperChunk {
    tex: GLuint,           // GL_TEXTURE_2D_ARRAY for color palette indices
    normal_tex: GLuint,    // GL_TEXTURE_2D_ARRAY for packed normals (RGB8)
    vao: GLuint,           // VAO with cube geometry + per-instance data
    instance_vbo: GLuint,  // VBO for VoxelInstanceData
    instances: u32

}

struct ChildNodeCache {
    nodes: HashMap<octree2::OctreeNode, [octree2::OctreeNode; 8]>,
    access: HashMap<octree2::OctreeNode, u64>,
    generation: u64,
}

impl ChildNodeCache {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            access: HashMap::new(),
            generation: 0,
        }
    }

    fn get(&mut self, node: &octree2::OctreeNode) -> &[octree2::OctreeNode; 8] {
        self.nodes.entry(node.clone()).or_insert_with(|| node.get_child_nodes())
    }

    fn get_and_track(&mut self, node: &octree2::OctreeNode) -> &[octree2::OctreeNode; 8] {
        self.access.insert(node.clone(), self.generation);
        self.get(node)
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.access.clear();
        self.generation = 0;
    }

    fn evict(&mut self, max_age_frames: u64) {
        let threshold = self.generation.saturating_sub(max_age_frames);
        let nodes = &mut self.nodes;
        self.access.retain(|k, &mut gen| {
            if gen >= threshold {
                true
            } else {
                nodes.remove(k);
                false
            }
        });
    }
}

/// Repack all chunks from z-major 4×4×4 into 16×4×N and upload in one call.
unsafe fn upload_chunks(tex: u32, chunks: &[VoxelChunk]) {
    let n = chunks.len();
    if n == 0 { return; }
    let mut buf = vec![0u16; 64 * n];
    for (layer, chunk) in chunks.iter().enumerate() {
        let base = layer * 64;
        for z in 0..4usize {
            for y in 0..4usize {
                for x in 0..4usize {
                    buf[base + y * 16 + (x + z * 4)] = chunk.voxels[z * 16 + y * 4 + x];
                }
            }
        }
    }
    gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex);
    gl::TexSubImage3D(
        gl::TEXTURE_2D_ARRAY, 0,
        0, 0, 0,
        16, 4, n as i32,
        gl::RED, gl::UNSIGNED_SHORT,
        buf.as_ptr() as *const _,
    );
}

/// Repack normals from z-major 4×4×4 into 16×4×N (RGBA) and upload.
unsafe fn upload_normals(tex: u32, chunks: &[VoxelChunk]) {
    let n = chunks.len();
    if n == 0 { return; }
    let mut buf = vec![0u8; 64 * 4 * n]; // 4 bytes per voxel (RGBA)
    for (layer, chunk) in chunks.iter().enumerate() {
        let base = layer * 64 * 4;
        for z in 0..4usize {
            for y in 0..4usize {
                for x in 0..4usize {
                    let src = z * 16 + y * 4 + x;
                    let dst = (y * 16 + (x + z * 4)) * 4;
                    buf[base + dst]     = chunk.normals[src][0];
                    buf[base + dst + 1] = chunk.normals[src][1];
                    buf[base + dst + 2] = chunk.normals[src][2];
                    buf[base + dst + 3] = chunk.normals[src][3];
                }
            }
        }
    }
    gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex);
    gl::TexSubImage3D(
        gl::TEXTURE_2D_ARRAY, 0,
        0, 0, 0,
        16, 4, n as i32,
        gl::RGBA, gl::UNSIGNED_BYTE,
        buf.as_ptr() as *const _,
    );
}

fn get_superchunk(node: octree2::OctreeNode, _center: Vec3, _size: f32, palette: &mut Palette, cube_vbo: GLuint, min_size: f32, cache: &mut ChildNodeCache) -> SuperChunk {

    // First pass: collect all chunks and instance data (no GL calls yet)
    let mut chunks: Vec<VoxelChunk> = Vec::new();
    let mut layer: Vec<VoxelInstanceData> = Vec::new();

    let mut stack: Vec<octree2::OctreeNode> = vec![node];
    while let Some(n) = stack.pop() {
        match &n {
            octree2::OctreeNode::Empty => {}
            octree2::OctreeNode::Node { center, size, sdf } => {
                if *size <= min_size {
                    if let Some(chunk) = voxelize_node(*center, *size, sdf, palette) {
                        let half = size / 2.0;
                        layer.push(VoxelInstanceData {
                            chunk_pos: [center.x - half, center.y - half, center.z - half],
                            atlas_layer: chunks.len() as u32,
                            chunk_size: *size,
                        });
                        chunks.push(chunk);
                    }
                } else {
                    let children = cache.get(&n);
                    for child in children.iter() {
                        stack.push(child.clone());
                    }
                }
            }
        }
    }

    let num_layers = chunks.len().max(1) as i32;
    //println!("Layer len: {}", chunks.len());

    // Create texture with exact size needed and upload all chunks in one go
    let mut tex: GLuint = 0;
    let mut normal_tex: GLuint = 0;
    unsafe {
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex);
        gl::TexStorage3D(gl::TEXTURE_2D_ARRAY, 1, gl::R16, 16, 4, num_layers);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_BASE_LEVEL, 0);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAX_LEVEL, 0);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        upload_chunks(tex, &chunks);

        gl::GenTextures(1, &mut normal_tex);
        gl::BindTexture(gl::TEXTURE_2D_ARRAY, normal_tex);
        gl::TexStorage3D(gl::TEXTURE_2D_ARRAY, 1, gl::RGBA8, 16, 4, num_layers);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_BASE_LEVEL, 0);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAX_LEVEL, 0);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        upload_normals(normal_tex, &chunks);
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
        tex,
        normal_tex,
        vao: sc_vao,
        instance_vbo: sc_instance_vbo,
        instances: layer.len() as u32
    }
}

unsafe fn create_lowres_fbo(w: u32, h: u32) -> (gl::types::GLuint, gl::types::GLuint, gl::types::GLuint) {
    let mut fbo = 0u32;
    let mut color = 0u32;
    let mut depth = 0u32;

    gl::GenFramebuffers(1, &mut fbo);
    gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);

    gl::GenTextures(1, &mut color);
    gl::BindTexture(gl::TEXTURE_2D, color);
    gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA8 as i32, w as i32, h as i32, 0, gl::RGBA, gl::UNSIGNED_BYTE, std::ptr::null());
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
    gl::FramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, color, 0);

    gl::GenRenderbuffers(1, &mut depth);
    gl::BindRenderbuffer(gl::RENDERBUFFER, depth);
    gl::RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH_COMPONENT24, w as i32, h as i32);
    gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, depth);

    gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
    (fbo, color, depth)
}

unsafe fn resize_lowres_fbo(fbo: &mut gl::types::GLuint, color: &mut gl::types::GLuint, depth: &mut gl::types::GLuint, w: u32, h: u32) {
    gl::DeleteFramebuffers(1, fbo);
    gl::DeleteTextures(1, color);
    gl::DeleteRenderbuffers(1, depth);
    let (f, c, d) = create_lowres_fbo(w, h);
    *fbo = f;
    *color = c;
    *depth = d;
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

    // --- Low-res FBO for pixelated rendering ---
    const RESOLUTION_SCALE: u32 = 4; // render at 1/3 resolution
    let (init_w, init_h) = window.get_framebuffer_size();
    let (mut render_w, mut render_h) = (
        (init_w as u32 / RESOLUTION_SCALE).max(1),
        (init_h as u32 / RESOLUTION_SCALE).max(1),
    );
    let (mut fbo, mut fbo_color, mut fbo_depth) = unsafe {
        create_lowres_fbo(render_w, render_h)
    };

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
    let mut octree2 = build_octree(&sdf, ROOT_SIZE);
    let mut palette_colors =0;
    
    let mut node_instance_lookup = HashMap::new();
    let mut child_cache = ChildNodeCache::new();

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
            unsafe { gl::GetUniformLocation(id, CString::new("uNormalAtlas").unwrap().as_ptr()) },
        )
    };

    // Cursor shader + geometry (3-axis crosshair made of 6 lines)
    let cursor_prog = {
        let vs = compile_gl_shader(CURSOR_VERTEX_SHADER_SRC, gl::VERTEX_SHADER);
        let fs = compile_gl_shader(CURSOR_FRAGMENT_SHADER_SRC, gl::FRAGMENT_SHADER);
        let id = link_program(vs, fs);
        (
            id,
            unsafe { gl::GetUniformLocation(id, CString::new("uViewProj").unwrap().as_ptr()) },
            unsafe { gl::GetUniformLocation(id, CString::new("uCenter").unwrap().as_ptr()) },
            unsafe { gl::GetUniformLocation(id, CString::new("uSize").unwrap().as_ptr()) },
        )
    };
    let cursor_vao = unsafe {
        let lines: [f32; 18] = [
            -1.0, 0.0, 0.0,   1.0, 0.0, 0.0,  // X axis
             0.0,-1.0, 0.0,   0.0, 1.0, 0.0,  // Y axis
             0.0, 0.0,-1.0,   0.0, 0.0, 1.0,  // Z axis
        ];
        let mut vao = 0u32;
        let mut vbo = 0u32;
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);
        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(gl::ARRAY_BUFFER, (lines.len() * 4) as isize, lines.as_ptr() as *const _, gl::STATIC_DRAW);
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 12, ptr::null());
        gl::EnableVertexAttribArray(0);
        gl::BindVertexArray(0);
        vao
    };
    let mut cursor_hit: Option<Vec3> = None;

    // Camera state
    let mut cam_pos = Vec3::new(0.0, 0.0, -60.0);
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.0;
    let mut last_cursor = (0.0f64, 0.0f64);
    let mut first_mouse = true;
    let move_speed = 5.0 * 1.0f32;
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
                            sdf = sdf.insert_2(new_sphere);
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
                            node_instance_lookup.clear();
                            child_cache.clear();
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

                    if let Some((_dist, hit_pos)) = fast_cast_ray(&octree2, &mut child_cache.nodes, cam_pos, ray_dir, 2000.0, 32.0) {
                        //sdf.print_layout(0);
                        sdf = sdf.subtract(Sdf::aabb(hit_pos /*- ray_dir * 5.0*/, Vec3::new(5.0, 5.0, 5.0))
                                      .with_color(Color::rgb(1.0, 1.0, 1.0)));
                        //sdf = sdf.optimize_bounds();
                        sdf_dirty = true;
                        println!("Subtracted sphere at {}", hit_pos);
                    }
                }
                glfw::WindowEvent::FramebufferSize(w, h) => unsafe {
                    render_w = (w as u32 / RESOLUTION_SCALE).max(1);
                    render_h = (h as u32 / RESOLUTION_SCALE).max(1);
                    resize_lowres_fbo(&mut fbo, &mut fbo_color, &mut fbo_depth, render_w, render_h);
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
            let _reused_count = 0u32;
            octree2 = OctreeNode2::get_node(Vec3::ZERO, ROOT_SIZE, &sdf);
            
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
        let proj = mat4::perspective(fovy, aspect, 0.1, 6000.0);
        let vp = mat4::mul(&proj, &view);

        // Cast ray from camera center for cursor
        cursor_hit = fast_cast_ray(&octree2, &mut child_cache.nodes, cam_pos, dir, 2000.0, 64.0).map(|(_, hit)| hit);

        unsafe {
            // Bind low-res FBO
            gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
            gl::Viewport(0, 0, render_w as i32, render_h as i32);

            // Sky background
            gl::ClearColor(0.25, 0.35, 0.55, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);

            gl::BindVertexArray(vao);

                const MAX_CHUNKS_PER_FRAME: u32 = 1;
                let frustum = Frustum::from_vp(&vp);
                let mut to_render = vec![];
                let mut chunks_generated = 0u32;
                child_cache.generation += 1;
                {
                    let mut oct_stack: Vec<octree2::OctreeNode> = vec![octree2.clone()];
                    while let Some(node) = oct_stack.pop() {
                    match node {
                        octree2::OctreeNode::Empty => {}

                        octree2::OctreeNode::Node { center, size, ..} => {
                            if frustum.cull_aabb(center, size / 2.0) { continue; }

                            let dist = (center - cam_pos).length();

                            let lod = calculate_lod(dist, 3.0 * 2000.0, 10) + 1;
                            if size <= 64.0 * (lod as f32) {
                                if node_instance_lookup.contains_key(&node) {
                                    to_render.push(node);
                                    continue;
                                }

                                if chunks_generated < MAX_CHUNKS_PER_FRAME {
                                    let copy = node.clone();
                                    let chunk = get_superchunk(node, center, size, &mut palette, vbo, (lod * 4) as f32, &mut child_cache);
                                    node_instance_lookup.insert(copy.clone(), chunk);
                                    to_render.push(copy);
                                    chunks_generated += 1;
                                    continue;
                                }
                                /*
                                // Budget exhausted — try coarser LOD
                                let coarse_lod = (lod + 1).max(2);
                                let coarse_min = (coarse_lod * 4) as f32;
                                let copy = node.clone();
                                let chunk = get_superchunk(node, center, size, &mut palette, vbo, coarse_min, &mut child_cache);
                                node_instance_lookup.insert(copy.clone(), chunk);
                                to_render.push(copy);*/
                                continue;
                            }
                            
                            let children = child_cache.get_and_track(&node);
                            // Recurse into children front-to-back
                            let near = ((cam_pos.x > center.x) as usize)
                                | (((cam_pos.y > center.y) as usize) << 1)
                                | (((cam_pos.z > center.z) as usize) << 2);
                            for &mask in &[7usize, 6, 5, 3, 4, 2, 1, 0] {
                                let child = &children[near ^ mask];
                                match child {
                                    octree2::OctreeNode::Node {..} => {
                                        oct_stack.push(child.clone());
                                    }
                                    octree2::OctreeNode::Empty => {}
                                }
                            }
                        }
                    }
                }
            }

            //println!("Rendering: {}", to_render.len());
            // Voxel rendering with LOD: per-frame octree traversal
            if !to_render.is_empty() {

                if palette_colors != palette.colors.len() {
                    palette_colors = palette.colors.len();
                    println!("update palette! {}", palette_colors);
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

                
                // Upload instance data and draw
                gl::UseProgram(voxel_prog.0);
                gl::UniformMatrix4fv(voxel_prog.1, 1, gl::FALSE, vp.as_ptr());
                gl::Uniform1i(voxel_prog.2, 0);  // brick atlas on texture unit 0
                gl::Uniform1i(voxel_prog.4, 1);  // palette on texture unit 1
                gl::Uniform1i(voxel_prog.5, 2);  // normal atlas on texture unit 2
                gl::Uniform3f(voxel_prog.3, cam_pos.x, cam_pos.y, cam_pos.z);

                // Bind palette texture to unit 1
                gl::ActiveTexture(gl::TEXTURE1);
                gl::BindTexture(gl::TEXTURE_1D, palette_tex);
                gl::ActiveTexture(gl::TEXTURE0);

                for (_i, node) in to_render.iter().enumerate() {
                    if let Some(sc) = node_instance_lookup.get(&node) {

                        gl::BindBuffer(gl::ARRAY_BUFFER, sc.instance_vbo);
                        // Brick atlas on unit 0
                        gl::ActiveTexture(gl::TEXTURE0);
                        gl::BindTexture(gl::TEXTURE_2D_ARRAY, sc.tex);
                        // Normal atlas on unit 2
                        gl::ActiveTexture(gl::TEXTURE2);
                        gl::BindTexture(gl::TEXTURE_2D_ARRAY, sc.normal_tex);
                        gl::BindVertexArray(sc.vao);
                        gl::DrawArraysInstanced(gl::TRIANGLES, 0, 36, sc.instances as GLsizei);
                    }
                }
                gl::ActiveTexture(gl::TEXTURE0);
                gl::BindVertexArray(0);
            }

            // Draw 3D cursor crosshair at hit point
            if let Some(hit) = cursor_hit {
                gl::UseProgram(cursor_prog.0);
                gl::UniformMatrix4fv(cursor_prog.1, 1, gl::FALSE, vp.as_ptr());
                gl::Uniform3f(cursor_prog.2, hit.x, hit.y, hit.z);
                gl::Uniform1f(cursor_prog.3, 2.0);
                gl::BindVertexArray(cursor_vao);
                gl::Disable(gl::DEPTH_TEST);
                gl::DrawArrays(gl::LINES, 0, 6);
                gl::Enable(gl::DEPTH_TEST);
                gl::BindVertexArray(0);
            }

            // Blit low-res FBO to screen with nearest filtering (pixelated)
            let (fw, fh) = window.get_framebuffer_size();
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, fbo);
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, 0);
            gl::BlitFramebuffer(
                0, 0, render_w as i32, render_h as i32,
                0, 0, fw, fh,
                gl::COLOR_BUFFER_BIT, gl::NEAREST,
            );
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        }

        window.swap_buffers();

        // FPS counter
        fps_frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(fps_last_time).as_secs_f64();
        if elapsed >= 1.0 {
            let recent = child_cache.access.values().filter(|&&gen| child_cache.generation - gen < fps_frame_count as u64).count();
            println!("FPS: {:.1}  child_cache: {} cached, {} recent", fps_frame_count as f64 / elapsed, child_cache.nodes.len(), recent);
            println!("CAMERA: {}", cam_pos);

            // Evict entries not accessed in the last 2 seconds worth of frames
            child_cache.evict(fps_frame_count as u64 * 2);

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
