use crate::sdf::{Coloring, DistanceFieldEnum, Primitive};
use std::rc::Rc;

/// Compiles a `DistanceFieldEnum` tree into a GLSL fragment shader that
/// ray-marches the SDF and shades the surface based on the color information
/// embedded in the tree.
///
/// The generated shader expects these uniforms:
///   - `u_resolution`: viewport size in pixels
///   - `u_camera_pos`: camera world position
///   - `u_camera_dir`: camera forward direction (normalized)
///   - `u_camera_up`: camera up direction (normalized)
pub fn compile_sdf_shader(sdf: &DistanceFieldEnum) -> String {
    let mut ctx = CompilerCtx::new();
    let dist_body = ctx.compile_distance(sdf);
    let color_body = ctx.compile_color(sdf);

    let helpers = &ctx.helpers;
    let param_count = ctx.param_count;

    let params_decl = if param_count > 0 {
        format!("uniform float u_params[{}];\nfloat P(int i) {{ return u_params[i]; }}\n", param_count)
    } else {
        String::new()
    };

    format!(
        r#"#version 330 core

out vec4 FragColor;

uniform vec2 u_resolution;
uniform vec3 u_camera_pos;
uniform vec3 u_camera_dir;
uniform vec3 u_camera_up;
{params_decl}
{helpers}

int g_closest_type;
int g_closest_idx;
float g_second_d;

float sdf_distance(vec3 p) {{
{dist_body}
}}

vec3 sdf_color(vec3 p) {{
{color_body}
}}

vec3 calc_normal(vec3 p) {{
    float e = 0.001;
    return normalize(vec3(
        sdf_distance(p + vec3(e, 0.0, 0.0)) - sdf_distance(p - vec3(e, 0.0, 0.0)),
        sdf_distance(p + vec3(0.0, e, 0.0)) - sdf_distance(p - vec3(0.0, e, 0.0)),
        sdf_distance(p + vec3(0.0, 0.0, e)) - sdf_distance(p - vec3(0.0, 0.0, e))
    ));
}}

void main() {{
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;

    vec3 right = normalize(cross(u_camera_up, u_camera_dir));
    vec3 up = cross(u_camera_dir, right);
    vec3 rd = normalize(uv.x * right + uv.y * up + 1.0 * u_camera_dir);

    vec3 ro = u_camera_pos;

    float t = 0.0;
    float max_dist = 500.0;
    bool hit = false;
    float iterations = 0.0;
    for (int i = 0; i < 256; i++) {{
        vec3 p = ro + rd * t;
        float d = sdf_distance(p);
        if (d < 0.001) {{
            hit = true;
            break;
        }}
        t += d;
        iterations += 1.0;
        if (t > max_dist) break;
    }}

    if (hit) {{
        vec3 p = ro + rd * t;
        vec3 n = calc_normal(p);
        vec3 base_color = sdf_color(p);

        // Simple directional lighting
        vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
        float diff = max(dot(n, light_dir), 0.0);
        float ambient = 0.15;
        vec3 color = base_color * (ambient + diff * 0.85);

        //FragColor = vec4(color, 1.0);
    }} else {{
        // Sky gradient
        float sky = 0.5 + 0.5 * rd.y;
        //FragColor = vec4(mix(vec3(0.1, 0.1, 0.15), vec3(0.4, 0.6, 0.9), sky), 1.0);
    }}
    iterations /= 10.0;
    FragColor=vec4(iterations, iterations, iterations, 1.0);

}}
"#
    )
}

struct CompilerCtx {
    helpers: String,
    next_id: u32,
    param_count: usize,
    has_smooth_subtract: bool,
    has_noise: bool,
    has_sphere_dist: bool,
    has_aabb_dist: bool,
    has_param3: bool,
    has_sphere_ray_intersect: bool,
}

impl CompilerCtx {
    fn new() -> Self {
        CompilerCtx {
            helpers: String::new(),
            next_id: 0,
            param_count: 0,
            has_smooth_subtract: false,
            has_noise: false,
            has_sphere_dist: false,
            has_aabb_dist: false,
            has_param3: false,
            has_sphere_ray_intersect: false,
        }
    }

    fn fresh_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn param(&mut self) -> String {
        let idx = self.param_count;
        self.param_count += 1;
        format!("P({})", idx)
    }

    /// Consume n params at once, return the starting index.
    fn params(&mut self, n: usize) -> usize {
        let idx = self.param_count;
        self.param_count += n;
        idx
    }

    /// Return (left, right) children of an Add in normalized order by topology hash,
    /// so that Add(A, B) and Add(B, A) produce identical shader code.
    fn ordered_add_children(add: &crate::sdf::Add) -> (&Rc<DistanceFieldEnum>, &Rc<DistanceFieldEnum>) {
        if add.left.topology_hash() <= add.right.topology_hash() {
            (&add.left, &add.right)
        } else {
            (&add.right, &add.left)
        }
    }

    fn ensure_sphere_dist(&mut self) {
        if !self.has_sphere_dist {
            self.has_sphere_dist = true;
            self.helpers.push_str(
                r#"
float sphere_dist(vec3 p, int i) {
    return length(p - vec3(P(i), P(i+1), P(i+2))) - P(i+3);
}

"#,
            );
        }
    }

    fn ensure_aabb_dist(&mut self) {
        if !self.has_aabb_dist {
            self.has_aabb_dist = true;
            self.helpers.push_str(
                r#"
float aabb_dist(vec3 p, int i) {
    vec3 d = abs(p - vec3(P(i), P(i+1), P(i+2))) - vec3(P(i+3), P(i+4), P(i+5));
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}
"#,
            );
        }
    }

    fn ensure_param3(&mut self) {
        if !self.has_param3 {
            self.has_param3 = true;
            self.helpers.push_str(
                r#"
vec3 param3(int i) {
    return vec3(P(i), P(i+1), P(i+2));
}
"#,
            );
        }
    }

    fn ensure_smooth_subtract(&mut self) {
        if !self.has_smooth_subtract {
            self.has_smooth_subtract = true;
            self.helpers.push_str(
                r#"
float smooth_subtract(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d1, -d2, h) + k * h * (1.0 - h);
}
"#,
            );
        }
    }

    fn ensure_noise(&mut self) {
        if !self.has_noise {
            self.has_noise = true;
            self.helpers.push_str(
                r#"
float hash3(vec3 p, float seed) {
    p += seed;
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p, p.yxz + 19.19);
    return fract((p.x + p.y) * p.z);
}

float value_noise(vec3 p, float seed) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash3(i + vec3(0,0,0), seed);
    float b = hash3(i + vec3(1,0,0), seed);
    float c = hash3(i + vec3(0,1,0), seed);
    float d = hash3(i + vec3(1,1,0), seed);
    float e = hash3(i + vec3(0,0,1), seed);
    float ff = hash3(i + vec3(1,0,1), seed);
    float g = hash3(i + vec3(0,1,1), seed);
    float h = hash3(i + vec3(1,1,1), seed);
    return mix(
        mix(mix(a, b, f.x), mix(c, d, f.x), f.y),
        mix(mix(e, ff, f.x), mix(g, h, f.x), f.y),
        f.z
    );
}

float fbm_noise(vec3 p, float seed) {
    float v = 0.0;
    v += value_noise(p * 1.0, seed) * 0.5;
    v += value_noise(p * 0.25, seed) * 0.25;
    v += value_noise(p * 4.0, seed) * 0.25;
    return v;
}
"#,
            );
        }
    }

    fn ensure_sphere_ray_intersect(&mut self) {
        if !self.has_sphere_ray_intersect {
            self.has_sphere_ray_intersect = true;
            self.helpers.push_str(
                r#"
float sphere_ray_intersect(vec3 ro, vec3 rd, int i) {
    vec3 oc = ro - vec3(P(i), P(i+1), P(i+2));
    float r = P(i+3);
    float b = dot(oc, rd);
    float c = dot(oc, oc) - r * r;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float sq = sqrt(disc);
    float t0 = -b - sq;
    if (t0 > 0.001) return t0;
    float t1 = -b + sq;
    if (t1 > 0.001) return t1;
    return -1.0;
}
"#,
            );
        }
    }

    /// Compile the distance function body.
    fn compile_distance(&mut self, sdf: &DistanceFieldEnum) -> String {
        let mut lines = String::new();
        let expr = self.emit_distance(sdf, &mut lines);
        lines.push_str(&format!("    return {expr};\n"));
        lines
    }

    /// Recursively emit GLSL for the distance computation. Appends variable
    /// declarations to `lines` and returns the final expression string.
    fn emit_distance(&mut self, sdf: &DistanceFieldEnum, lines: &mut String) -> String {
        match sdf {
            DistanceFieldEnum::Empty => {
                lines.push_str("    g_closest_type = -1;\n");
                lines.push_str("    g_second_d = 1.0e10;\n");
                "1.0e10".to_string()
            }

            DistanceFieldEnum::Primitive(prim) => match prim {
                Primitive::Sphere(_) => {
                    self.ensure_sphere_dist();
                    let idx = self.params(4);
                    let id = self.fresh_id();
                    lines.push_str(&format!("    float _pd{id} = sphere_dist(p, {idx});\n"));
                    lines.push_str(&format!("    g_closest_type = 0;\n"));
                    lines.push_str(&format!("    g_closest_idx = {idx};\n"));
                    lines.push_str("    g_second_d = 1.0e10;\n");
                    format!("_pd{id}")
                }
                Primitive::Aabb(_) => {
                    self.ensure_aabb_dist();
                    let idx = self.params(6);
                    let id = self.fresh_id();
                    lines.push_str(&format!("    float _pd{id} = aabb_dist(p, {idx});\n"));
                    lines.push_str(&format!("    g_closest_type = 1;\n"));
                    lines.push_str(&format!("    g_closest_idx = {idx};\n"));
                    lines.push_str("    g_second_d = 1.0e10;\n");
                    format!("_pd{id}")
                }
            },

            DistanceFieldEnum::Coloring(_, inner) => {
                self.emit_distance(inner, lines)
            }

            DistanceFieldEnum::Add(add) => {
                self.ensure_sphere_dist();
                let id = self.fresh_id();
                let (child_a, child_b) = Self::ordered_add_children(add);
                // Bounding sphere early-out
                let bidx = self.params(4);
                lines.push_str(&format!(
                    "    float _bd{id} = sphere_dist(p, {bidx});\n"
                ));
                lines.push_str(&format!("    float _da{id};\n"));
                lines.push_str(&format!("    if (_bd{id} > P({r})) {{\n", r = bidx + 3));
                lines.push_str(&format!("        _da{id} = _bd{id};\n"));
                lines.push_str("        g_closest_type = -1;\n");
                lines.push_str("        g_second_d = 1.0e10;\n");
                lines.push_str("    } else {\n");

                // Left child
                let mut left_lines = String::new();
                let left_expr = self.emit_distance(child_a, &mut left_lines);

                // Right child
                let mut right_lines = String::new();
                let right_expr = self.emit_distance(child_b, &mut right_lines);

                let sid = self.fresh_id();

                // Emit left lines (indented)
                for line in left_lines.lines() {
                    lines.push_str("    ");
                    lines.push_str(line);
                    lines.push('\n');
                }
                // Save left globals
                lines.push_str(&format!("        int _lt{sid} = g_closest_type;\n"));
                lines.push_str(&format!("        int _li{sid} = g_closest_idx;\n"));
                lines.push_str(&format!("        float _ls{sid} = g_second_d;\n"));
                lines.push_str(&format!("        float _ld{sid} = {left_expr};\n"));

                // Emit right lines (indented)
                for line in right_lines.lines() {
                    lines.push_str("    ");
                    lines.push_str(line);
                    lines.push('\n');
                }
                lines.push_str(&format!("        float _rd{sid} = {right_expr};\n"));

                // Pick winner
                lines.push_str(&format!("        if (_ld{sid} < _rd{sid}) {{\n"));
                lines.push_str(&format!("            g_closest_type = _lt{sid};\n"));
                lines.push_str(&format!("            g_closest_idx = _li{sid};\n"));
                lines.push_str(&format!("            g_second_d = min(_rd{sid}, _ls{sid});\n"));
                lines.push_str(&format!("        }} else {{\n"));
                lines.push_str(&format!("            g_second_d = min(_ld{sid}, g_second_d);\n"));
                lines.push_str(&format!("        }}\n"));
                lines.push_str(&format!("        _da{id} = min(_ld{sid}, _rd{sid});\n"));

                lines.push_str("    }\n");
                format!("_da{id}")
            }

            DistanceFieldEnum::Subtract(sub) => {
                self.ensure_smooth_subtract();

                // Left child (sets globals)
                let mut left_lines = String::new();
                let left_expr = self.emit_distance(&sub.left, &mut left_lines);
                lines.push_str(&left_lines);

                // Save left globals
                let sid = self.fresh_id();
                lines.push_str(&format!("    int _lt{sid} = g_closest_type;\n"));
                lines.push_str(&format!("    int _li{sid} = g_closest_idx;\n"));
                lines.push_str(&format!("    float _ls{sid} = g_second_d;\n"));

                // Right child (overwrites globals)
                let mut right_lines = String::new();
                let right_expr = self.emit_distance(&sub.subtract, &mut right_lines);
                lines.push_str(&right_lines);

                // Restore left globals
                lines.push_str(&format!("    g_closest_type = _lt{sid};\n"));
                lines.push_str(&format!("    g_closest_idx = _li{sid};\n"));
                lines.push_str(&format!("    g_second_d = _ls{sid};\n"));

                let k = self.param();
                let id = self.fresh_id();
                lines.push_str(&format!(
                    "    float _ds{id} = smooth_subtract({left_expr}, {right_expr}, {k});\n"
                ));
                format!("_ds{id}")
            }
        }
    }

    /// Compile the color function body.
    fn compile_color(&mut self, sdf: &DistanceFieldEnum) -> String {
        let mut lines = String::new();
        let expr = self.emit_color(sdf, &mut lines);
        lines.push_str(&format!("    return {expr};\n"));
        lines
    }

    /// Recursively emit GLSL for color computation. Returns a vec3 expression.
    fn emit_color(&mut self, sdf: &DistanceFieldEnum, lines: &mut String) -> String {
        match sdf {
            DistanceFieldEnum::Empty => "vec3(0.0)".to_string(),

            DistanceFieldEnum::Primitive(_) => "vec3(1.0, 0.0, 0.0)".to_string(),

            DistanceFieldEnum::Coloring(coloring, _inner) => match coloring {
                Coloring::SolidColor(_) => {
                    self.ensure_param3();
                    let idx = self.params(3);
                    format!("param3({})", idx)
                }
                Coloring::Gradient(_) => {
                    self.ensure_param3();
                    let id = self.fresh_id();
                    let idx = self.params(12);
                    lines.push_str(&format!(
                        "    vec3 _gp1_{id} = param3({});\n", idx
                    ));
                    lines.push_str(&format!(
                        "    vec3 _gp2_{id} = param3({});\n", idx + 3
                    ));
                    format!(
                        "mix(param3({}), param3({}), clamp(dot(_gp2_{id} - _gp1_{id}, p - _gp1_{id}) / dot(_gp1_{id} - _gp2_{id}, _gp1_{id} - _gp2_{id}), 0.0, 1.0))",
                        idx + 6, idx + 9,
                    )
                }
                Coloring::Noise(_) => {
                    self.ensure_noise();
                    self.ensure_param3();
                    let id = self.fresh_id();
                    let idx = self.params(7);
                    lines.push_str(&format!(
                        "    float _nv{id} = fbm_noise(p, P({}));\n", idx
                    ));
                    format!(
                        "mix(param3({}), param3({}), _nv{id})",
                        idx + 1, idx + 4,
                    )
                }
            },

            DistanceFieldEnum::Add(add) => {
                let (child_a, child_b) = Self::ordered_add_children(add);
                // Need distance of each branch to pick the closer one's color
                let ld = {
                    let mut dl = String::new();
                    let expr = self.emit_distance(child_a, &mut dl);
                    lines.push_str(&dl);
                    expr
                };
                let rd = {
                    let mut dr = String::new();
                    let expr = self.emit_distance(child_b, &mut dr);
                    lines.push_str(&dr);
                    expr
                };
                let id = self.fresh_id();
                lines.push_str(&format!("    float _ld{id} = {ld};\n"));
                lines.push_str(&format!("    float _rd{id} = {rd};\n"));

                let lc = {
                    let mut cl = String::new();
                    let expr = self.emit_color(child_a, &mut cl);
                    lines.push_str(&cl);
                    expr
                };
                let rc = {
                    let mut cr = String::new();
                    let expr = self.emit_color(child_b, &mut cr);
                    lines.push_str(&cr);
                    expr
                };

                let cid = self.fresh_id();
                lines.push_str(&format!(
                    "    vec3 _ca{cid} = (_ld{id} < _rd{id}) ? {lc} : {rc};\n"
                ));
                format!("_ca{cid}")
            }

            DistanceFieldEnum::Subtract(sub) => {
                self.emit_color(&sub.left, lines)
            }
        }
    }
}

pub struct CompiledBlockShader {
    pub source: String,
    pub param_count: usize,
    /// vec4s per instance in the UBO: 2 (render_min, render_max) + ceil(param_count/4)
    pub stride: usize,
}

/// Compiles a `DistanceFieldEnum` tree into a GLSL fragment shader for instanced
/// block rendering. The shader reads per-instance data (render bounds, params) from
/// a UBO indexed by `v_data_base` (set from `gl_InstanceID` in the vertex shader).
///
/// Uniforms:
///   - `u_camera_pos`: camera world position
///   - `u_vp`: combined view-projection matrix (for depth output)
///   - `InstanceData` UBO: vec4 array containing per-instance render bounds + params
pub fn compile_block_sdf_shader(sdf: &DistanceFieldEnum) -> CompiledBlockShader {
    let mut ctx = CompilerCtx::new();
    let dist_body = ctx.compile_distance(sdf);
    let color_body = ctx.compile_color(sdf);
    ctx.ensure_sphere_ray_intersect();

    let helpers = &ctx.helpers;
    let param_count = ctx.param_count;
    let stride = 2 + (param_count + 3) / 4;

    let params_decl = if param_count > 0 {
        "flat in int v_data_base;\nlayout(std140) uniform InstanceData {\n    vec4 u_data[4096];\n};\nfloat P(int i) { return u_data[v_data_base + 2 + i/4][i & 3]; }\n".to_string()
    } else {
        String::new()
    };

    let source = format!(
        r#"#version 330 core

in vec3 v_world_pos;
flat in vec3 v_block_min;
flat in vec3 v_block_max;
out vec4 FragColor;

uniform vec3 u_camera_pos;
uniform mat4 u_vp;
{params_decl}
{helpers}

int g_closest_type;
int g_closest_idx;
float g_second_d;

float sdf_distance(vec3 p) {{
{dist_body}
}}

vec3 sdf_color(vec3 p) {{
{color_body}
}}

vec3 calc_normal(vec3 p) {{
    float e = 0.001;
    return normalize(vec3(
        sdf_distance(p + vec3(e, 0.0, 0.0)) - sdf_distance(p - vec3(e, 0.0, 0.0)),
        sdf_distance(p + vec3(0.0, e, 0.0)) - sdf_distance(p - vec3(0.0, e, 0.0)),
        sdf_distance(p + vec3(0.0, 0.0, e)) - sdf_distance(p - vec3(0.0, 0.0, e))
    ));
}}

void main() {{
    vec3 ro = u_camera_pos;
    vec3 rd = normalize(v_world_pos - ro);

    // Ray-AABB intersection
    vec3 t1 = (v_block_min - ro) / rd;
    vec3 t2 = (v_block_max - ro) / rd;
    vec3 tmin_v = min(t1, t2);
    vec3 tmax_v = max(t1, t2);
    float t_enter = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    float t_exit = min(min(tmax_v.x, tmax_v.y), tmax_v.z);

    if (t_enter > t_exit || t_exit < 0.0) discard;

    float t = max(t_enter - 0.01, 0.0);
    float iterations = 0.0;
    bool hit = false;
    for (int i = 0; i < 128; i++) {{
        iterations += 1.0;
    vec3 p = ro + rd * t;
        float d = sdf_distance(p);
        if (d < 0.001) {{
            hit = true;
            break;
        }}

        if (g_closest_type == 0) {{
            float t_hit = sphere_ray_intersect(ro, rd, g_closest_idx);
            if (t_hit > t + 0.001 && t_hit - t < g_second_d) {{
                // Jump directly to sphere surface
                t = t_hit * 1.001;
                continue;
            }} else if (t_hit < 0.0) {{
                // Ray misses sphere — skip by second closest
                t += g_second_d;
                if (t > t_exit) break;
                continue;
            }}
        }}

        t += d;
        if (t > t_exit) break;
    }}

    if (!hit) discard;

    vec3 p = ro + rd * t;
    vec3 n = calc_normal(p);
    vec3 base_color = sdf_color(p);

    // Simple directional lighting
    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(n, light_dir), 0.0);
    float ambient = 0.15;
    vec3 color = base_color * (ambient + diff * 0.85);

    FragColor = vec4(color, 1.0);
    //iterations /= 5.0;
    //FragColor= vec4(iterations, iterations, iterations, 1.0);
    // Write depth from hit point
    vec4 clip = u_vp * vec4(p, 1.0);

    gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
}}
"#
    );

    CompiledBlockShader { source, param_count, stride }
}

/// Collect all parameter values from an SDF tree in the same order as the
/// compiler emits `u_params[N]` references. This must mirror the traversal
/// order of `emit_distance` followed by `emit_color`.
pub fn collect_block_sdf_params(sdf: &DistanceFieldEnum) -> Vec<f32> {
    let mut params = Vec::new();
    collect_distance_params(sdf, &mut params);
    collect_color_params(sdf, &mut params);
    params
}

fn collect_distance_params(sdf: &DistanceFieldEnum, params: &mut Vec<f32>) {
    match sdf {
        DistanceFieldEnum::Empty => {}

        DistanceFieldEnum::Primitive(prim) => match prim {
            Primitive::Sphere(s) => {
                params.push(s.center.x);
                params.push(s.center.y);
                params.push(s.center.z);
                params.push(s.radius);
            }
            Primitive::Aabb(a) => {
                params.push(a.center.x);
                params.push(a.center.y);
                params.push(a.center.z);
                params.push(a.radius.x);
                params.push(a.radius.y);
                params.push(a.radius.z);
            }
        },

        DistanceFieldEnum::Coloring(_, inner) => {
            collect_distance_params(inner, params);
        }

        DistanceFieldEnum::Add(add) => {
            let (child_a, child_b) = CompilerCtx::ordered_add_children(add);
            // Bounding sphere params
            params.push(add.bounds.center.x);
            params.push(add.bounds.center.y);
            params.push(add.bounds.center.z);
            params.push(add.bounds.radius);
            // Children in normalized order
            collect_distance_params(child_a, params);
            collect_distance_params(child_b, params);
        }

        DistanceFieldEnum::Subtract(sub) => {
            collect_distance_params(&sub.left, params);
            collect_distance_params(&sub.subtract, params);
            params.push(sub.k);
        }
    }
}

fn collect_color_params(sdf: &DistanceFieldEnum, params: &mut Vec<f32>) {
    match sdf {
        DistanceFieldEnum::Empty => {}

        DistanceFieldEnum::Primitive(_) => {}

        DistanceFieldEnum::Coloring(coloring, _inner) => match coloring {
            Coloring::SolidColor(c) => {
                params.push(c.r);
                params.push(c.g);
                params.push(c.b);
            }
            Coloring::Gradient(g) => {
                params.push(g.p1.x);
                params.push(g.p1.y);
                params.push(g.p1.z);
                params.push(g.p2.x);
                params.push(g.p2.y);
                params.push(g.p2.z);
                params.push(g.c1.r);
                params.push(g.c1.g);
                params.push(g.c1.b);
                params.push(g.c2.r);
                params.push(g.c2.g);
                params.push(g.c2.b);
            }
            Coloring::Noise(n) => {
                params.push(n.seed as f32);
                params.push(n.c1.r);
                params.push(n.c1.g);
                params.push(n.c1.b);
                params.push(n.c2.r);
                params.push(n.c2.g);
                params.push(n.c2.b);
            }
        },

        DistanceFieldEnum::Add(add) => {
            let (child_a, child_b) = CompilerCtx::ordered_add_children(add);
            // Re-collect distance params for both children (mirrors emit_color's
            // call to emit_distance on each child)
            collect_distance_params(child_a, params);
            collect_distance_params(child_b, params);
            // Then collect color params for each child
            collect_color_params(child_a, params);
            collect_color_params(child_b, params);
        }

        DistanceFieldEnum::Subtract(sub) => {
            collect_color_params(&sub.left, params);
        }
    }
}

/// Format an f32 for GLSL output (always includes a decimal point).
fn ff(v: f32) -> String {
    if v == v.floor() && v.abs() < 1e7 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::sdf::*;
    use crate::vec3::Vec3;

    #[test]
    fn test_compile_single_sphere() {
        let sdf: DistanceFieldEnum =
            Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0).color(Color::RED);
        let shader = compile_sdf_shader(&sdf);
        assert!(shader.contains("sdf_distance"));
        assert!(shader.contains("sdf_color"));
        assert!(shader.contains("sphere_dist(p"));
        println!("{}", shader);
    }

    #[test]
    fn test_compile_add() {
        let s1: DistanceFieldEnum =
            Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0).color(Color::RED);
        let s2: DistanceFieldEnum =
            Sphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0).color(Color::BLUE);
        let sdf = Add::new2(s1, s2);
        let shader = compile_sdf_shader(&sdf);
        // Should contain bounding sphere check and min()
        assert!(shader.contains("_bd"));
        assert!(shader.contains("min("));
        println!("{}", shader);
    }

    #[test]
    fn test_compile_subtract() {
        let s1: DistanceFieldEnum =
            Sphere::new(Vec3::new(0.0, 0.0, 0.0), 2.0).color(Color::RED);
        let s2: DistanceFieldEnum = Sphere::new(Vec3::new(1.0, 0.0, 0.0), 1.0).into();
        let sdf = s1.subtract(s2);
        let shader = compile_sdf_shader(&sdf);
        assert!(shader.contains("smooth_subtract"));
        println!("{}", shader);
    }

    #[test]
    fn test_compile_complex_scene() {
        let sdf = build_test_solid();
        let shader = compile_sdf_shader(&sdf);
        assert!(shader.contains("#version 330 core"));
        assert!(shader.contains("FragColor"));
        println!("{}", shader);
    }

    #[test]
    fn test_block_shader_uses_params() {
        let sdf: DistanceFieldEnum =
            Sphere::new(Vec3::new(1.0, 2.0, 3.0), 4.0).color(Color::RED);
        let compiled = compile_block_sdf_shader(&sdf);
        assert!(compiled.source.contains("InstanceData"));
        assert!(compiled.source.contains("u_data"));
        assert!(compiled.source.contains("P("));
        assert!(compiled.param_count > 0);
        assert!(compiled.stride >= 2);
        // Should not contain literal values for the sphere
        assert!(!compiled.source.contains("1.0, 2.0, 3.0"));
        println!("param_count = {}, stride = {}", compiled.param_count, compiled.stride);
        println!("{}", compiled.source);
    }

    #[test]
    fn test_collect_params_matches_count() {
        let sdf: DistanceFieldEnum =
            Sphere::new(Vec3::new(1.0, 2.0, 3.0), 4.0).color(Color::RED);
        let compiled = compile_block_sdf_shader(&sdf);
        let params = collect_block_sdf_params(&sdf);
        assert_eq!(params.len(), compiled.param_count,
            "collected params len ({}) != compiled param_count ({})",
            params.len(), compiled.param_count);
        // Sphere: cx, cy, cz, r = 4 distance params
        // SolidColor: r, g, b = 3 color params
        // Total = 7
        assert_eq!(params.len(), 7);
        assert_eq!(params[0], 1.0); // cx
        assert_eq!(params[1], 2.0); // cy
        assert_eq!(params[2], 3.0); // cz
        assert_eq!(params[3], 4.0); // radius
    }

    #[test]
    fn test_collect_params_add() {
        let s1: DistanceFieldEnum =
            Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0).color(Color::RED);
        let s2: DistanceFieldEnum =
            Sphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0).color(Color::BLUE);
        let sdf = Add::new2(s1, s2);
        let compiled = compile_block_sdf_shader(&sdf);
        let params = collect_block_sdf_params(&sdf);
        assert_eq!(params.len(), compiled.param_count,
            "collected params len ({}) != compiled param_count ({})",
            params.len(), compiled.param_count);
    }

    #[test]
    fn test_collect_params_subtract() {
        let s1: DistanceFieldEnum =
            Sphere::new(Vec3::new(0.0, 0.0, 0.0), 2.0).color(Color::RED);
        let s2: DistanceFieldEnum = Sphere::new(Vec3::new(1.0, 0.0, 0.0), 1.0).into();
        let sdf = s1.subtract(s2);
        let compiled = compile_block_sdf_shader(&sdf);
        let params = collect_block_sdf_params(&sdf);
        assert_eq!(params.len(), compiled.param_count,
            "collected params len ({}) != compiled param_count ({})",
            params.len(), compiled.param_count);
    }

    #[test]
    fn test_block_shader_has_globals() {
        let s1: DistanceFieldEnum =
            Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0).color(Color::RED);
        let s2: DistanceFieldEnum =
            Sphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0).color(Color::BLUE);
        let sdf = Add::new2(s1, s2);
        let compiled = compile_block_sdf_shader(&sdf);
        assert!(compiled.source.contains("g_closest_type"));
        assert!(compiled.source.contains("g_closest_idx"));
        assert!(compiled.source.contains("g_second_d"));
        assert!(compiled.source.contains("sphere_ray_intersect"));
        println!("{}", compiled.source);
    }
}
