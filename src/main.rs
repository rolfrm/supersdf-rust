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
use rand::{Rng, SeedableRng, rngs::StdRng};

// ---------- Octree constants ----------
const MIN_NODE_SIZE: f32 = 0.5;
const ROOT_SIZE: f32 = 4000.0;

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

// ---------- Debug box shader (flat color, alpha) ----------
const DEBUG_BOX_FRAG_SRC: &str = r#"
#version 330 core
uniform vec3 u_color;
out vec4 FragColor;
void main() {
    FragColor = vec4(u_color, 0.12);
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

fn mat4_block_model_box(block_min: Vec3, block_size: Vec3) -> Mat4 {
    [
        block_size.x, 0.0, 0.0, 0.0,
        0.0, block_size.y, 0.0, 0.0,
        0.0, 0.0, block_size.z, 0.0,
        block_min.x, block_min.y, block_min.z, 1.0,
    ]
}

// ---------- Frustum culling ----------

struct Frustum {
    planes: [[f32; 4]; 6],
}

impl Frustum {
    /// Extract frustum planes from a column-major view-projection matrix.
    fn from_vp(vp: &Mat4) -> Frustum {
        // Row i of column-major matrix: vp[i], vp[i+4], vp[i+8], vp[i+12]
        let raw = [
            [vp[3]+vp[0], vp[7]+vp[4], vp[11]+vp[8],  vp[15]+vp[12]], // Left
            [vp[3]-vp[0], vp[7]-vp[4], vp[11]-vp[8],  vp[15]-vp[12]], // Right
            [vp[3]+vp[1], vp[7]+vp[5], vp[11]+vp[9],  vp[15]+vp[13]], // Bottom
            [vp[3]-vp[1], vp[7]-vp[5], vp[11]-vp[9],  vp[15]-vp[13]], // Top
            [vp[3]+vp[2], vp[7]+vp[6], vp[11]+vp[10], vp[15]+vp[14]], // Near
            [vp[3]-vp[2], vp[7]-vp[6], vp[11]-vp[10], vp[15]-vp[14]], // Far
        ];
        let mut planes = [[0.0f32; 4]; 6];
        for i in 0..6 {
            let len = (raw[i][0] * raw[i][0] + raw[i][1] * raw[i][1] + raw[i][2] * raw[i][2]).sqrt();
            if len > 0.0 {
                let inv = 1.0 / len;
                planes[i] = [raw[i][0] * inv, raw[i][1] * inv, raw[i][2] * inv, raw[i][3] * inv];
            }
        }
        Frustum { planes }
    }

    /// Returns true if a cube (center + half-extent) is entirely outside the frustum.
    fn cull_aabb(&self, center: Vec3, half: f32) -> bool {
        for p in &self.planes {
            // Effective radius: max projection of the box onto the plane normal
            let r = half * (p[0].abs() + p[1].abs() + p[2].abs());
            let dist = p[0] * center.x + p[1] * center.y + p[2] * center.z + p[3];
            if dist + r < 0.0 {
                return true;
            }
        }
        false
    }
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
    //println!("compile: {}", frag_src);
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

// ---------- Octree ----------

#[derive(Clone)]
enum OctreeNode {
    Leaf {
        center: Vec3,
        size: f32,
        render_min: Vec3,
        render_max: Vec3,
        optimized_sdf: Rc<DistanceFieldEnum>,
        topology_hash: u64,
        params: Vec<f32>,
    },
    Branch {
        center: Vec3,
        size: f32,
        optimized_sdf: Rc<DistanceFieldEnum>,
        children: [Option<Rc<OctreeNode>>; 8],
    },
    Empty,
}

struct Octree {
    root: OctreeNode,
    programs: HashMap<u64, BlockProgram>,
}

/// Return the center offset for child octant `i` given `half` = child_size / 2.
fn octant_offset(i: usize, half: f32) -> Vec3 {
    Vec3::new(
        if i & 1 != 0 { half } else { -half },
        if i & 2 != 0 { half } else { -half },
        if i & 4 != 0 { half } else { -half },
    )
}


impl Octree {
    fn new() -> Octree {
        Octree {
            root: OctreeNode::Empty,
            programs: HashMap::new(),
        }
    }

    

    fn build_node(
        center: Vec3,
        size: f32,
        sdf: &DistanceFieldEnum,
        old_node: &OctreeNode,
        cache: &mut HashSet<DistanceFieldEnum>,
        to_compile: &mut HashMap<u64, Rc<DistanceFieldEnum>>,
        reused_count: &mut u32,
    ) -> OctreeNode {
        let optimized = sdf.optimized_for_block(center, size, cache);

        // Empty check
        if matches!(optimized.as_ref(), DistanceFieldEnum::Empty) {
            return OctreeNode::Empty;
        }

        // AABB vs AABB check — tighter cull than sphere bounds
        let bounds = optimized.calculate_aabb_bounds();
        if bounds.is_finite() && !bounds.overlaps_aabb(center, size / 2.0) {
            return OctreeNode::Empty;
        }

        let half_diag = size * 0.5 * 1.7320508;
        if optimized.distance(center) > half_diag {
            return OctreeNode::Empty;
        }

        // Change detection: if the optimized SDF matches old node's, reuse subtree
        match old_node {
            OctreeNode::Leaf { optimized_sdf, .. }
            | OctreeNode::Branch { optimized_sdf, .. }
                if optimized.equals( optimized_sdf)  =>
            {
                // Collect all hashes from the reused subtree so we don't delete their programs
                Self::collect_hashes(old_node, to_compile);
                *reused_count += 1;
                return old_node.clone();
            }
            _ => {}
        }

        let prim_count = optimized.count_primitives_up_to(6);

        // Leaf condition: stop subdividing at min size or <=6 primitives
        if (size <= MIN_NODE_SIZE && prim_count <= 6) || prim_count <= 6 {
            let hash = optimized.topology_hash();
            to_compile.entry(hash).or_insert_with(|| optimized.clone());
            let params = sdf_compiler::collect_block_sdf_params(&optimized);
            let half = size / 2.0;
            let cell_min = center - Vec3::new(half, half, half);
            let cell_max = center + Vec3::new(half, half, half);
            let content = optimized.calculate_aabb_bounds();
            let render_min = Vec3::new(cell_min.x.max(content.min.x), cell_min.y.max(content.min.y), cell_min.z.max(content.min.z));
            let render_max = Vec3::new(cell_max.x.min(content.max.x), cell_max.y.min(content.max.y), cell_max.z.min(content.max.z));
            return OctreeNode::Leaf {
                center,
                size,
                render_min,
                render_max,
                optimized_sdf: optimized,
                topology_hash: hash,
                params,
            };
        }

        // Subdivide into 8 children
        let child_size = size / 2.0;
        let children: [Option<Rc<OctreeNode>>; 8] = std::array::from_fn(|i| {
            let child_center = center + octant_offset(i, child_size / 2.0);
            let old_child = match old_node {
                OctreeNode::Branch { children, .. } => children[i]
                    .as_ref()
                    .map(|rc| rc.as_ref())
                    .unwrap_or(&OctreeNode::Empty),
                _ => &OctreeNode::Empty,
            };
            let child = Self::build_node(
                child_center,
                child_size,
                &optimized,
                old_child,
                cache,
                to_compile,
                reused_count,
                );

            match child {
                OctreeNode::Empty => None,
                node => Some(Rc::new(node)),
            }
        });

        // Count non-empty children
        let non_empty_count = children.iter().filter(|c| c.is_some()).count();

        // All children empty → this branch is empty
        if non_empty_count == 0 {
            return OctreeNode::Empty;
        }

        // Only one child with simple SDF → collapse to a leaf
        // (must use parent's center/size/optimized so change detection matches next rebuild)
        if non_empty_count == 1 && prim_count <= 3 {
            let hash = optimized.topology_hash();
            to_compile.entry(hash).or_insert_with(|| optimized.clone());
            let params = sdf_compiler::collect_block_sdf_params(&optimized);
            let half = size / 2.0;
            let cell_min = center - Vec3::new(half, half, half);
            let cell_max = center + Vec3::new(half, half, half);
            let content = optimized.calculate_aabb_bounds();
            let render_min = Vec3::new(cell_min.x.max(content.min.x), cell_min.y.max(content.min.y), cell_min.z.max(content.min.z));
            let render_max = Vec3::new(cell_max.x.min(content.max.x), cell_max.y.min(content.max.y), cell_max.z.min(content.max.z));
            return OctreeNode::Leaf {
                center,
                size,
                render_min,
                render_max,
                optimized_sdf: optimized,
                topology_hash: hash,
                params,
            };
        }

        // Collapse: if every non-empty child is a leaf with the same topology hash,
        // merge back into one leaf at the parent size.
        let first_node = children.iter().flatten().find_map(|c| match c.as_ref() {
            OctreeNode::Leaf {optimized_sdf, .. } => Some(optimized_sdf.clone()),
            _ => None,
        });
        if let Some(fh) = first_node {
            let can_collapse = prim_count < 5
                && children.iter().all(|c| match c {
                    Some(rc) => matches!(rc.as_ref(), OctreeNode::Leaf { optimized_sdf, .. } if optimized_sdf.equals(&fh)),
                    None => true,
                });
            if can_collapse {
                let hash = optimized.topology_hash();
                to_compile.entry(hash).or_insert_with(|| optimized.clone());
                let params = sdf_compiler::collect_block_sdf_params(&optimized);
                let half = size / 2.0;
                let cell_min = center - Vec3::new(half, half, half);
                let cell_max = center + Vec3::new(half, half, half);
                let content = optimized.calculate_aabb_bounds();
                let render_min = Vec3::new(cell_min.x.max(content.min.x), cell_min.y.max(content.min.y), cell_min.z.max(content.min.z));
                let render_max = Vec3::new(cell_max.x.min(content.max.x), cell_max.y.min(content.max.y), cell_max.z.min(content.max.z));
                return OctreeNode::Leaf {
                    center,
                    size,
                    render_min,
                    render_max,
                    optimized_sdf: optimized,
                    topology_hash: hash,
                    params,
                };
            }
        }

        OctreeNode::Branch {
            center,
            size,
            optimized_sdf: optimized,
            children,
        }
    }

    /// Collect topology hashes from a subtree (for reused nodes whose programs we must keep).
    fn collect_hashes(node: &OctreeNode, to_compile: &mut HashMap<u64, Rc<DistanceFieldEnum>>) {
        match node {
            OctreeNode::Leaf { topology_hash, optimized_sdf, .. } => {
                to_compile.entry(*topology_hash).or_insert_with(|| optimized_sdf.clone());
            }
            OctreeNode::Branch { children, .. } => {
                for child in children.iter().flatten() {
                    Self::collect_hashes(child, to_compile);
                }
            }
            OctreeNode::Empty => {}
        }
    }

    fn rebuild(&mut self, sdf: &DistanceFieldEnum) {
        let mut cache = HashSet::new();
        let mut to_compile: HashMap<u64, Rc<DistanceFieldEnum>> = HashMap::new();
        let mut reused_count = 0u32;

        let root_center = Vec3::new(0.0, 0.0, 0.0);
        let new_root = Self::build_node(
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

        self.root = new_root;
        let leaf_count = Self::count_leaves(&self.root);
        println!(
            "Octree rebuild: {} leaves, {} unique ({} compiled, {} cached), {} subtrees reused",
            leaf_count, to_compile.len(), compiled, already_cached, reused_count
        );
    }

    fn count_leaves(node: &OctreeNode) -> u32 {
        match node {
            OctreeNode::Empty => 0,
            OctreeNode::Leaf { .. } => 1,
            OctreeNode::Branch { children, .. } => {
                children.iter().flatten().map(|c| Self::count_leaves(c)).sum()
            }
        }
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

    let field_size = 200;
    
    for i in (-field_size..field_size).step_by(10) {
        for j in (-field_size..field_size).step_by(10) {
            let x = i as f32 + rng.gen_range(-2.0..2.0);
            let z = j as f32 + rng.gen_range(-2.0..2.0);
            let y = -20.0 + rng.gen_range(-2.0..2.0) + j as f32 * 0.2;
            let r = rng.gen_range(8.0..10.0);
            let color = Color::rgb(
                rng.gen_range(0.2..0.22),
                rng.gen_range(0.9..1.0),
                rng.gen_range(0.2..0.22),
            );
            sdf = sdf.insert_2(DistanceFieldEnum::sphere(Vec3::new(x, y, z), r).colored(color));
        }
    }

    sdf.optimize_bounds()
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

    let mut octree = Octree::new();

    // Debug box shader for visualizing octree nodes
    let debug_prog = {
        let vs = compile_gl_shader(BLOCK_VERTEX_SHADER_SRC, gl::VERTEX_SHADER);
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

        // Recompile shaders if SDF changed
        if sdf_dirty {
            sdf_dirty = false;
            octree.rebuild(&sdf);
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
        let proj = mat4_perspective(fovy, aspect, 0.1, 4000.0);
        let vp = mat4_mul(&proj, &view);

        unsafe {
            // Sky background
            gl::ClearColor(0.25, 0.35, 0.55, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);

            gl::BindVertexArray(vao);

            // Frustum culling + front-to-back traversal
            let frustum = Frustum::from_vp(&vp);
            let mut leaf_stack: Vec<&OctreeNode> = vec![&octree.root];
            while let Some(node) = leaf_stack.pop() {
                match node {
                    OctreeNode::Empty => {}
                    OctreeNode::Leaf { center, size, render_min, render_max, topology_hash, params, .. } => {
                        if frustum.cull_aabb(*center, *size / 2.0) { continue; }
                        if let Some(prog) = octree.programs.get(topology_hash) {
                            let block_size = *render_max - *render_min;
                            let model = mat4_block_model_box(*render_min, block_size);

                            gl::UseProgram(prog.id);
                            gl::UniformMatrix4fv(prog.u_vp, 1, gl::FALSE, vp.as_ptr());
                            gl::UniformMatrix4fv(prog.u_model, 1, gl::FALSE, model.as_ptr());
                            gl::Uniform3f(prog.u_camera_pos, cam_pos.x, cam_pos.y, cam_pos.z);
                            gl::Uniform3f(prog.u_block_min, render_min.x, render_min.y, render_min.z);
                            gl::Uniform3f(prog.u_block_max, render_max.x, render_max.y, render_max.z);

                            if !params.is_empty() {
                                gl::Uniform1fv(prog.u_params, params.len() as i32, params.as_ptr());
                            }

                            gl::DrawArrays(gl::TRIANGLES, 0, 36);
                        }
                    }
                    OctreeNode::Branch { center: bc, size, children, .. } => {
                        if frustum.cull_aabb(*bc, *size / 2.0) { continue; }
                        // Determine nearest octant to camera using sign bits
                        let near = ((cam_pos.x > bc.x) as usize)
                            | (((cam_pos.y > bc.y) as usize) << 1)
                            | (((cam_pos.z > bc.z) as usize) << 2);
                        // Push farthest first (popcount 3→0) so nearest pops first from stack
                        for &mask in &[7usize, 6, 5, 3, 4, 2, 1, 0] {
                            if let Some(ref child) = children[near ^ mask] {
                                leaf_stack.push(child.as_ref());
                            }
                        }
                    }
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
                            let model = mat4_block_model_box(*render_min, block_size);
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
    }

    octree.cleanup();
    unsafe {
        gl::DeleteProgram(debug_prog.0);
        gl::DeleteVertexArrays(1, &vao);
        gl::DeleteBuffers(1, &vbo);
    }
}
