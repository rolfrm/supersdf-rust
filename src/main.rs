extern crate gl;
extern crate glfw;
mod octree2;

use supersdf::color::*;
use supersdf::sdf::*;
use supersdf::sdf_compiler;
use supersdf::vec3::Vec3;
use supersdf::mat4::{self, Frustum};
use supersdf::octree::{OctreeNode, ROOT_SIZE};
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

// ---------- Block vertex shader (instanced, UBO) ----------
fn block_vertex_shader_src(stride: usize) -> String {
    format!(
        r#"#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 u_vp;
layout(std140) uniform InstanceData {{
    vec4 u_data[4096];
}};
out vec3 v_world_pos;
flat out vec3 v_block_min;
flat out vec3 v_block_max;
flat out int v_data_base;
void main() {{
    int base = gl_InstanceID * {stride};
    vec3 render_min = u_data[base].xyz;
    vec3 render_max = u_data[base + 1].xyz;
    vec3 world = render_min + aPos * (render_max - render_min);
    v_world_pos = world;
    v_block_min = render_min;
    v_block_max = render_max;
    v_data_base = base;
    gl_Position = u_vp * vec4(world, 1.0);
}}
"#
    )
}


// ---------- Debug box vertex shader (non-instanced) ----------
const DEBUG_VERTEX_SHADER_SRC: &str = r#"
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

// ---------- Debug box shader (flat color, alpha) ----------
const DEBUG_BOX_FRAG_SRC: &str = r#"
#version 330 core
uniform vec3 u_color;
out vec4 FragColor;
void main() {
    FragColor = vec4(u_color, 0.12);
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

// ---------- Block program ----------

struct BlockProgram {
    id: GLuint,
    u_vp: GLint,
    u_camera_pos: GLint,
    /// vec4s per instance in the UBO
    stride: usize,
    /// max instances per draw call (4096 / stride)
    max_instances: usize,
}

fn build_block_program(frag_src: &str, param_count: usize) -> Option<BlockProgram> {
    let stride = 2 + (param_count + 3) / 4;
    let max_instances = 4096 / stride;
    let vs_src = block_vertex_shader_src(stride);

    let vs = compile_gl_shader(&vs_src, gl::VERTEX_SHADER);
    if vs == 0 { return None; }

    let fs = compile_gl_shader(frag_src, gl::FRAGMENT_SHADER);
    if fs == 0 {
        unsafe { gl::DeleteShader(vs); }
        return None;
    }

    let id = link_program(vs, fs);
    if id == 0 { return None; }

    unsafe {
        // Bind the InstanceData UBO to binding point 0
        let block_name = CString::new("InstanceData").unwrap();
        let ubo_idx = gl::GetUniformBlockIndex(id, block_name.as_ptr());
        if ubo_idx != gl::INVALID_INDEX {
            gl::UniformBlockBinding(id, ubo_idx, 0);
        }

        Some(BlockProgram {
            id,
            u_vp: gl::GetUniformLocation(id, CString::new("u_vp").unwrap().as_ptr()),
            u_camera_pos: gl::GetUniformLocation(id, CString::new("u_camera_pos").unwrap().as_ptr()),
            stride,
            max_instances,
        })
    }
}

// ---------- GL Octree wrapper ----------

struct Octree {
    root: OctreeNode,
    programs: HashMap<u64, BlockProgram>,
}

impl Octree {
    fn new() -> Octree {
        Octree {
            root: OctreeNode::Empty,
            programs: HashMap::new(),
        }
    }

    fn rebuild(&mut self, sdf: &DistanceFieldEnum) {
        let mut cache = HashSet::new();
        let mut to_compile: HashMap<u64, Rc<DistanceFieldEnum>> = HashMap::new();
        let mut reused_count = 0u32;

        let root_center = Vec3::new(0.0, 0.0, 0.0);
        let new_root = OctreeNode::build_node(
            root_center,
            ROOT_SIZE,
            sdf,
            &self.root,
            &mut cache,
            &mut to_compile,
            &mut reused_count,
        );

        // Delete programs no longer needed
        let needed: HashSet<u64> = to_compile.keys().copied().collect();
        self.programs.retain(|hash, prog| {
            if needed.contains(hash) {
                true
            } else {
                unsafe { gl::DeleteProgram(prog.id); }
                false
            }
        });

        // Compile only new unique programs
        let mut compiled = 0u32;
        let mut already_cached = 0u32;
        for (hash, optimized) in &to_compile {
            if self.programs.contains_key(hash) {
                already_cached += 1;
            } else {
                let compiled_shader = sdf_compiler::compile_block_sdf_shader(optimized);
                match build_block_program(&compiled_shader.source, compiled_shader.param_count) {
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

        self.root = new_root;
        let leaf_count = OctreeNode::count_leaves(&self.root);
        println!(
            "Octree rebuild: {} leaves, {} unique ({} compiled, {} cached), {} subtrees reused",
            leaf_count, to_compile.len(), compiled, already_cached, reused_count
        );
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
    let mut rng = StdRng::seed_from_u64(42);
    
    let field_size = 1000;

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
    //sdf = sdf.insert_2(DistanceFieldEnum::aabb(Vec3::new(0.0, -2018.0, 0.0), Vec3::new(field_size as f32, 2000.0, field_size as f32)));
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

fn hash_sdf(sdf: &DistanceFieldEnum) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    sdf.hash(&mut hasher);
    hasher.finish()
}

type BrickCacheKey = ((i32, i32, i32, i32), u64); // (brick_key, sdf_hash)

/// Caches voxelized brick data across octree rebuilds.
/// Keyed by (spatial_key, sdf_hash) with equality check on the SDF to avoid hash collisions.
struct BrickCache {
    entries: HashMap<BrickCacheKey, (Rc<DistanceFieldEnum>, VoxelChunk)>,
}

impl BrickCache {
    fn new() -> Self {
        BrickCache { entries: HashMap::new() }
    }

    /// Look up a cached brick. Returns Some if the SDF matches (hash + equality).
    fn get(&self, bkey: (i32, i32, i32, i32), sdf_hash: u64, sdf: &DistanceFieldEnum) -> Option<&VoxelChunk> {
        if let Some((cached_sdf, chunk)) = self.entries.get(&(bkey, sdf_hash)) {
            if cached_sdf.as_ref() == sdf {
                return Some(chunk);
            }
        }
        None
    }

    /// Insert a voxelized brick into the cache.
    fn insert(&mut self, bkey: (i32, i32, i32, i32), sdf_hash: u64, sdf: &Rc<DistanceFieldEnum>, chunk: VoxelChunk) {
        self.entries.insert((bkey, sdf_hash), (sdf.clone(), chunk));
    }

    /// Remove entries not present in the given set of keys.
    fn retain_keys(&mut self, used_keys: &HashSet<BrickCacheKey>) {
        self.entries.retain(|k, _| used_keys.contains(k));
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

struct BrickInfo {
    chunk: VoxelChunk,
    center: Vec3,
    size: f32,
}

/// Walk the octree and voxelize all leaves (size=4) and branches up to MAX_LOD_SIZE.
/// Uses brick_cache to skip re-voxelization of unchanged nodes.
fn collect_all_bricks(
    node: &octree2::OctreeNode,
    palette: &mut Palette,
    brick_cache: &mut BrickCache,
) -> Vec<BrickInfo> {
    let mut bricks = Vec::new();
    let mut used_keys: HashSet<BrickCacheKey> = HashSet::new();
    let mut cache_hits = 0u32;
    let mut cache_misses = 0u32;
    let mut stack: Vec<&octree2::OctreeNode> = vec![node];

    while let Some(n) = stack.pop() {
        match n {
            octree2::OctreeNode::Empty => {}
            octree2::OctreeNode::Leaf { center, size, optimized_sdf } => {
                let bkey = brick_key(center, *size);
                let sdf_hash = hash_sdf(optimized_sdf);
                let cache_key = (bkey, sdf_hash);
                used_keys.insert(cache_key);

                if let Some(cached) = brick_cache.get(bkey, sdf_hash, optimized_sdf) {
                    bricks.push(BrickInfo { chunk: cached.clone(), center: *center, size: *size });
                    cache_hits += 1;
                } else if let Some(chunk) = voxelize_node(center, *size, optimized_sdf, palette) {
                    brick_cache.insert(bkey, sdf_hash, optimized_sdf, chunk.clone());
                    bricks.push(BrickInfo { chunk, center: *center, size: *size });
                    cache_misses += 1;
                }
            }
            octree2::OctreeNode::Branch { center, size, optimized_sdf, children } => {
                if *size <= MAX_LOD_SIZE {
                    let bkey = brick_key(center, *size);
                    let sdf_hash = hash_sdf(optimized_sdf);
                    let cache_key = (bkey, sdf_hash);
                    used_keys.insert(cache_key);

                    if let Some(cached) = brick_cache.get(bkey, sdf_hash, optimized_sdf) {
                        bricks.push(BrickInfo { chunk: cached.clone(), center: *center, size: *size });
                        cache_hits += 1;
                    } else if let Some(chunk) = voxelize_node(center, *size, optimized_sdf, palette) {
                        brick_cache.insert(bkey, sdf_hash, optimized_sdf, chunk.clone());
                        bricks.push(BrickInfo { chunk, center: *center, size: *size });
                        cache_misses += 1;
                    }
                }
                for child in children.iter().flatten() {
                    stack.push(child.as_ref());
                }
            }
        }
    }

    // Prune stale cache entries
    let old_size = brick_cache.entries.len();
    brick_cache.retain_keys(&used_keys);
    let pruned = old_size - brick_cache.entries.len();
    println!("Brick cache: {} hits, {} misses, {} pruned, {} total",
        cache_hits, cache_misses, pruned, brick_cache.entries.len());

    bricks
}

/// Key for brick lookup: quantized center position + size
fn brick_key(center: &Vec3, size: f32) -> (i32, i32, i32, i32) {
    (
        (center.x * 4.0).round() as i32,
        (center.y * 4.0).round() as i32,
        (center.z * 4.0).round() as i32,
        (size * 4.0).round() as i32,
    )
}

#[derive(Clone, Copy)]
struct BrickLocation {
    super_chunk_idx: usize,
    local_layer: u32,
    chunk_pos: [f32; 3],
    chunk_size: f32,
}

struct SuperChunk {
    tex: GLuint,           // GL_TEXTURE_2D_ARRAY, up to MAX_LAYERS_PER_TEXTURE layers
    vao: GLuint,           // VAO with cube geometry + per-instance data
    instance_vbo: GLuint,  // VBO for VoxelInstanceData
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

struct VoxelMap {
    octree: octree2::OctreeNode,
    super_chunks: Vec<SuperChunk>,
    brick_map: HashMap<(i32, i32, i32, i32), BrickLocation>,
    palette_tex: GLuint,
}

impl VoxelMap {
    /// Build voxel map from SDF: octree + bricks + super chunks + palette texture.
    /// `palette` and `brick_cache` persist across rebuilds for incremental updates.

    fn rebuild(&self,
        sdf: &DistanceFieldEnum,
        cube_vbo: GLuint,
        palette: &mut Palette,
        brick_cache: &mut BrickCache,
        ) -> VoxelMap {
         let mut cache = HashSet::new();
        let mut reused_count = 0u32;
        let octree = OctreeNode2::build_node(Vec3::ZERO, 2048.0, &sdf, &self.octree, &mut cache, &mut reused_count);
        println!("Reused: {}", reused_count);
        VoxelMap::build0(sdf, cube_vbo, palette, brick_cache, octree)
    }

    fn build(
        sdf: &DistanceFieldEnum,
        cube_vbo: GLuint,
        palette: &mut Palette,
        brick_cache: &mut BrickCache,
    ) -> VoxelMap {
        let octree = build_octree(sdf, 2048.0);
        VoxelMap::build0(sdf, cube_vbo, palette, brick_cache, octree)

    }

    
    fn build0(sdf: &DistanceFieldEnum,
             cube_vbo: GLuint,
             palette: &mut Palette,
             brick_cache: &mut BrickCache,
             octree: octree2::OctreeNode
             ) -> VoxelMap {
        let n_leaves = octree.count_leaves();
        println!("Collect bricks..");
        let all_bricks = collect_all_bricks(&octree, palette, brick_cache);
        let total = all_bricks.len();
        println!("Leaves: {}, total bricks (leaves + LOD): {}, palette: {} colors", n_leaves, total, palette.colors.len());

        // Build palette texture (1D, 256 entries, RGB)
        let mut palette_tex: GLuint = 0;
        unsafe {
            gl::GenTextures(1, &mut palette_tex);
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
        println!("regenerated palette");

        // Build super chunks
        let mut super_chunks: Vec<SuperChunk> = Vec::new();
        let mut brick_map: HashMap<(i32, i32, i32, i32), BrickLocation> = HashMap::new();
        let n_super = if total == 0 { 0 } else { (total + MAX_LAYERS_PER_TEXTURE - 1) / MAX_LAYERS_PER_TEXTURE };

        for sc_idx in 0..n_super {
            let start = sc_idx * MAX_LAYERS_PER_TEXTURE;
            let end = (start + MAX_LAYERS_PER_TEXTURE).min(total);
            let layer_count = end - start;

            let mut tex: GLuint = 0;
            unsafe {
                gl::GenTextures(1, &mut tex);
                gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex);
                gl::TexStorage3D(gl::TEXTURE_2D_ARRAY, 1, gl::R8, 16, 4, layer_count as i32);
                gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
                gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
                gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_BASE_LEVEL, 0);
                gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_MAX_LEVEL, 0);
                gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
                gl::TexParameteri(gl::TEXTURE_2D_ARRAY, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            }

            for i in start..end {
                let local_layer = (i - start) as u32;
                unsafe { upload_chunk(tex, local_layer, &all_bricks[i].chunk); }

                let b = &all_bricks[i];
                let half = b.size / 2.0;
                let chunk_pos = [b.center.x - half, b.center.y - half, b.center.z - half];
                let key = brick_key(&b.center, b.size);
                brick_map.insert(key, BrickLocation {
                    super_chunk_idx: sc_idx,
                    local_layer,
                    chunk_pos,
                    chunk_size: b.size,
                });
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
                    (layer_count * std::mem::size_of::<VoxelInstanceData>()) as GLsizeiptr,
                    ptr::null(),
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

            super_chunks.push(SuperChunk {
                tex,
                vao: sc_vao,
                instance_vbo: sc_instance_vbo,
            });
        }
        println!("Super chunks: {}", super_chunks.len());

        VoxelMap { octree, super_chunks, brick_map, palette_tex }

    }
    
    
    fn cleanup(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.palette_tex);
            for sc in &self.super_chunks {
                gl::DeleteVertexArrays(1, &sc.vao);
                gl::DeleteBuffers(1, &sc.instance_vbo);
                gl::DeleteTextures(1, &sc.tex);
            }
        }
        self.super_chunks.clear();
        self.brick_map.clear();
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

    // UBO for instanced rendering (per-instance data: render bounds + params)
    let mut ubo = 0u32;
    unsafe {
        gl::GenBuffers(1, &mut ubo);
    }

    let mut octree = Octree::new();

    // Debug box shader for visualizing octree nodes
    let debug_prog = {
        let vs = compile_gl_shader(DEBUG_VERTEX_SHADER_SRC, gl::VERTEX_SHADER);
        let fs = compile_gl_shader(DEBUG_BOX_FRAG_SRC, gl::FRAGMENT_SHADER);
        let id = link_program(vs, fs);
        (
            id,
            unsafe { gl::GetUniformLocation(id, CString::new("u_vp").unwrap().as_ptr()) },
            unsafe { gl::GetUniformLocation(id, CString::new("u_model").unwrap().as_ptr()) },
            unsafe { gl::GetUniformLocation(id, CString::new("u_color").unwrap().as_ptr()) },
        )
    };
    let mut show_debug_boxes = false;

    let mut palette = Palette::new();
    let mut brick_cache = BrickCache::new();
    let mut voxel_map = VoxelMap::build(&sdf, vbo, &mut palette, &mut brick_cache);

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
    let path_tracing_enabled = false;

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
                            show_debug_boxes = !show_debug_boxes;
                            println!("Debug boxes: {}", if show_debug_boxes { "ON" } else { "OFF" });
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
            //octree.rebuild(&sdf);
            voxel_map.cleanup();
            voxel_map = voxel_map.rebuild(&sdf, vbo, &mut palette, &mut brick_cache);
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

            if path_tracing_enabled {
                // Phase 1: Collect visible leaves, pack UBO data per topology group
                // Ordered by first-seen (front-to-back traversal), instances within
                // each group also stay in front-to-back order.
                let frustum = Frustum::from_vp(&vp);

                struct GroupData {
                    hash: u64,
                    ubo_data: Vec<f32>,
                    instance_count: usize,
                }
                let mut groups: Vec<GroupData> = Vec::new();
                let mut group_idx: HashMap<u64, usize> = HashMap::new();

                let mut leaf_stack: Vec<&OctreeNode> = vec![&octree.root];
                while let Some(node) = leaf_stack.pop() {
                    match node {
                        OctreeNode::Empty => {}
                        OctreeNode::Leaf { center, size, render_min, render_max, topology_hash, params, .. } => {
                            if frustum.cull_aabb(*center, *size / 2.0) { continue; }
                            if let Some(prog) = octree.programs.get(topology_hash) {
                                let fpi = prog.stride * 4; // floats per instance
                                let idx = *group_idx.entry(*topology_hash).or_insert_with(|| {
                                    let i = groups.len();
                                    groups.push(GroupData {
                                        hash: *topology_hash,
                                        ubo_data: Vec::new(),
                                        instance_count: 0,
                                    });
                                    i
                                });
                                let group = &mut groups[idx];
                                let base = group.ubo_data.len();
                                group.ubo_data.resize(base + fpi, 0.0);
                                // vec4[0]: render_min
                                group.ubo_data[base]     = render_min.x;
                                group.ubo_data[base + 1] = render_min.y;
                                group.ubo_data[base + 2] = render_min.z;
                                // vec4[1]: render_max
                                group.ubo_data[base + 4] = render_max.x;
                                group.ubo_data[base + 5] = render_max.y;
                                group.ubo_data[base + 6] = render_max.z;
                                // vec4[2..]: params packed into vec4s
                                for (j, &p) in params.iter().enumerate() {
                                    group.ubo_data[base + 8 + j] = p;
                                }
                                group.instance_count += 1;
                            }
                        }
                        OctreeNode::Branch { center: bc, size, children, .. } => {
                            if frustum.cull_aabb(*bc, *size / 2.0) { continue; }
                            let near = ((cam_pos.x > bc.x) as usize)
                                | (((cam_pos.y > bc.y) as usize) << 1)
                                | (((cam_pos.z > bc.z) as usize) << 2);
                            for &mask in &[7usize, 6, 5, 3, 4, 2, 1, 0] {
                                if let Some(ref child) = children[near ^ mask] {
                                    leaf_stack.push(child.as_ref());
                                }
                            }
                        }
                    }
                }

                // Phase 2: Draw groups in first-seen order (front-to-back)
                gl::BindVertexArray(vao);
                for group in &groups {
                    if let Some(prog) = octree.programs.get(&group.hash) {
                        gl::UseProgram(prog.id);
                        gl::UniformMatrix4fv(prog.u_vp, 1, gl::FALSE, vp.as_ptr());
                        gl::Uniform3f(prog.u_camera_pos, cam_pos.x, cam_pos.y, cam_pos.z);

                        let fpi = prog.stride * 4;
                        for batch_start in (0..group.instance_count).step_by(prog.max_instances) {
                            let batch_count = (group.instance_count - batch_start).min(prog.max_instances);
                            let float_offset = batch_start * fpi;
                            let byte_size = batch_count * fpi * std::mem::size_of::<f32>();

                            gl::BindBuffer(gl::UNIFORM_BUFFER, ubo);
                            gl::BufferData(
                                gl::UNIFORM_BUFFER,
                                byte_size as GLsizeiptr,
                                group.ubo_data[float_offset..].as_ptr() as *const _,
                                gl::STATIC_DRAW,
                            );
                            gl::BindBufferBase(gl::UNIFORM_BUFFER, 0, ubo);

                            gl::DrawArraysInstanced(gl::TRIANGLES, 0, 36, batch_count as GLsizei);
                        }
                    }
                }
            } // end path_tracing_enabled
            else {
            // Voxel rendering with LOD: per-frame octree traversal
            if !voxel_map.super_chunks.is_empty() {
                let frustum = Frustum::from_vp(&vp);

                // Per-super-chunk instance lists (rebuilt each frame)
                let mut sc_instances: Vec<Vec<VoxelInstanceData>> = (0..voxel_map.super_chunks.len())
                    .map(|_| Vec::new())
                    .collect();

                // Traverse octree with LOD early termination
                let mut lod_stack: Vec<&octree2::OctreeNode> = vec![&voxel_map.octree];
                while let Some(node) = lod_stack.pop() {
                    match node {
                        octree2::OctreeNode::Empty => {}
                        octree2::OctreeNode::Leaf { center, size, .. } => {
                            if frustum.cull_aabb(*center, *size / 2.0) { continue; }
                            let key = brick_key(center, *size);
                            if let Some(loc) = voxel_map.brick_map.get(&key) {
                                sc_instances[loc.super_chunk_idx].push(VoxelInstanceData {
                                    chunk_pos: loc.chunk_pos,
                                    atlas_layer: loc.local_layer,
                                    chunk_size: loc.chunk_size,
                                });
                            }
                        }
                        octree2::OctreeNode::Branch { center, size, children, .. } => {
                            if frustum.cull_aabb(*center, *size / 2.0) { continue; }

                            // LOD check: if far enough and we have a coarse brick, use it
                            let dist = (*center - cam_pos).length();
                            if *size <= MAX_LOD_SIZE && dist > *size * LOD_FACTOR {
                                let key = brick_key(center, *size);
                                if let Some(loc) = voxel_map.brick_map.get(&key) {
                                    sc_instances[loc.super_chunk_idx].push(VoxelInstanceData {
                                        chunk_pos: loc.chunk_pos,
                                        atlas_layer: loc.local_layer,
                                        chunk_size: loc.chunk_size,
                                    });
                                    continue; // skip children
                                }
                            }

                            // Recurse into children front-to-back
                            let near = ((cam_pos.x > center.x) as usize)
                                | (((cam_pos.y > center.y) as usize) << 1)
                                | (((cam_pos.z > center.z) as usize) << 2);
                            for &mask in &[7usize, 6, 5, 3, 4, 2, 1, 0] {
                                if let Some(ref child) = children[near ^ mask] {
                                    lod_stack.push(child.as_ref());
                                }
                            }
                        }
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
                gl::BindTexture(gl::TEXTURE_1D, voxel_map.palette_tex);
                gl::ActiveTexture(gl::TEXTURE0);

                for (i, instances) in sc_instances.iter().enumerate() {
                    if instances.is_empty() { continue; }
                    let sc = &voxel_map.super_chunks[i];
                    gl::BindBuffer(gl::ARRAY_BUFFER, sc.instance_vbo);
                    gl::BufferSubData(
                        gl::ARRAY_BUFFER,
                        0,
                        (instances.len() * std::mem::size_of::<VoxelInstanceData>()) as GLsizeiptr,
                        instances.as_ptr() as *const _,
                    );
                    gl::BindTexture(gl::TEXTURE_2D_ARRAY, sc.tex);
                    gl::BindVertexArray(sc.vao);
                    gl::DrawArraysInstanced(gl::TRIANGLES, 0, 36, instances.len() as GLsizei);
                }
                gl::BindVertexArray(0);
            }
            }
                

            // Debug pass: draw translucent boxes for each leaf node
            if show_debug_boxes {
                gl::Enable(gl::BLEND);
                gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
                gl::DepthMask(gl::FALSE);
                gl::UseProgram(debug_prog.0);
                gl::UniformMatrix4fv(debug_prog.1, 1, gl::FALSE, vp.as_ptr());

                let mut debug_stack: Vec<&OctreeNode> = vec![&octree.root];
                let mut leaf_idx: u32 = 0;
                while let Some(node) = debug_stack.pop() {
                    match node {
                        OctreeNode::Empty => {}
                        OctreeNode::Leaf { render_min, render_max, .. } => {
                            let block_size = *render_max - *render_min;
                            let model = mat4::block_model_box(*render_min, block_size);
                            gl::UniformMatrix4fv(debug_prog.2, 1, gl::FALSE, model.as_ptr());

                            // Color by hash of leaf index for variety
                            let r = ((leaf_idx * 97 + 31) % 255) as f32 / 255.0;
                            let g = ((leaf_idx * 53 + 73) % 255) as f32 / 255.0;
                            let b = ((leaf_idx * 179 + 17) % 255) as f32 / 255.0;
                            gl::Uniform3f(debug_prog.3, r, g, b);

                            gl::DrawArrays(gl::TRIANGLES, 0, 36);
                            leaf_idx += 1;
                        }
                        OctreeNode::Branch { children, .. } => {
                            for child in children.iter().flatten() {
                                debug_stack.push(child);
                            }
                        }
                    }
                }

                gl::DepthMask(gl::TRUE);
                gl::Disable(gl::BLEND);
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

    octree.cleanup();
    voxel_map.cleanup();
    unsafe {
        gl::DeleteProgram(debug_prog.0);
        gl::DeleteProgram(voxel_prog.0);
        gl::DeleteVertexArrays(1, &vao);
        gl::DeleteBuffers(1, &vbo);
        gl::DeleteBuffers(1, &ubo);
    }
}
