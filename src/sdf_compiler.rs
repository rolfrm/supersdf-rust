use crate::sdf::{Coloring, DistanceFieldEnum, Primitive};

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

    format!(
        r#"#version 330 core

out vec4 FragColor;

uniform vec2 u_resolution;
uniform vec3 u_camera_pos;
uniform vec3 u_camera_dir;
uniform vec3 u_camera_up;

{helpers}

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

    for (int i = 0; i < 256; i++) {{
        vec3 p = ro + rd * t;
        float d = sdf_distance(p);
        if (d < 0.001) {{
            hit = true;
            break;
        }}
        t += d;
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

        FragColor = vec4(color, 1.0);
    }} else {{
        // Sky gradient
        float sky = 0.5 + 0.5 * rd.y;
        FragColor = vec4(mix(vec3(0.1, 0.1, 0.15), vec3(0.4, 0.6, 0.9), sky), 1.0);
    }}
}}
"#
    )
}

struct CompilerCtx {
    helpers: String,
    next_id: u32,
    has_smooth_subtract: bool,
    has_noise: bool,
}

impl CompilerCtx {
    fn new() -> Self {
        CompilerCtx {
            helpers: String::new(),
            next_id: 0,
            has_smooth_subtract: false,
            has_noise: false,
        }
    }

    fn fresh_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
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
            DistanceFieldEnum::Empty => "1.0e10".to_string(),

            DistanceFieldEnum::Primitive(prim) => match prim {
                Primitive::Sphere(s) => {
                    format!(
                        "(length(p - vec3({}, {}, {})) - {})",
                        ff(s.center.x), ff(s.center.y), ff(s.center.z),
                        ff(s.radius)
                    )
                }
                Primitive::Aabb(a) => {
                    let id = self.fresh_id();
                    lines.push_str(&format!(
                        "    vec3 _bp{id} = abs(p - vec3({}, {}, {})) - vec3({}, {}, {});\n",
                        ff(a.center.x), ff(a.center.y), ff(a.center.z),
                        ff(a.radius.x), ff(a.radius.y), ff(a.radius.z),
                    ));
                    format!(
                        "(length(max(_bp{id}, 0.0)) + min(max(_bp{id}.x, max(_bp{id}.y, _bp{id}.z)), 0.0))"
                    )
                }
            },

            DistanceFieldEnum::Coloring(_, inner) => {
                self.emit_distance(inner, lines)
            }

            DistanceFieldEnum::Add(add) => {
                let id = self.fresh_id();
                let b = &add.bounds;
                // Bounding sphere early-out: if point is far from bounds,
                // skip evaluating both children and return bounds distance.
                lines.push_str(&format!(
                    "    float _bd{id} = length(p - vec3({}, {}, {})) - {};\n",
                    ff(b.center.x), ff(b.center.y), ff(b.center.z), ff(b.radius)
                ));
                lines.push_str(&format!("    float _da{id};\n"));
                lines.push_str(&format!("    if (_bd{id} > {}) {{\n", ff(b.radius)));
                lines.push_str(&format!("        _da{id} = _bd{id};\n"));
                lines.push_str("    } else {\n");

                let mut inner = String::new();
                let left = self.emit_distance(&add.left, &mut inner);
                let right = self.emit_distance(&add.right, &mut inner);
                for line in inner.lines() {
                    lines.push_str("    ");
                    lines.push_str(line);
                    lines.push('\n');
                }
                lines.push_str(&format!("        _da{id} = min({left}, {right});\n"));
                lines.push_str("    }\n");
                format!("_da{id}")
            }

            DistanceFieldEnum::Subtract(sub) => {
                self.ensure_smooth_subtract();
                let left = self.emit_distance(&sub.left, lines);
                let right = self.emit_distance(&sub.subtract, lines);
                let id = self.fresh_id();
                lines.push_str(&format!(
                    "    float _ds{id} = smooth_subtract({left}, {right}, {});\n",
                    ff(sub.k)
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
                Coloring::SolidColor(c) => {
                    format!("vec3({}, {}, {})", ff(c.r), ff(c.g), ff(c.b))
                }
                Coloring::Gradient(g) => {
                    let id = self.fresh_id();
                    lines.push_str(&format!(
                        "    vec3 _gp1_{id} = vec3({}, {}, {});\n",
                        ff(g.p1.x), ff(g.p1.y), ff(g.p1.z)
                    ));
                    lines.push_str(&format!(
                        "    vec3 _gp2_{id} = vec3({}, {}, {});\n",
                        ff(g.p2.x), ff(g.p2.y), ff(g.p2.z)
                    ));
                    format!(
                        "mix(vec3({}, {}, {}), vec3({}, {}, {}), clamp(dot(_gp2_{id} - _gp1_{id}, p - _gp1_{id}) / dot(_gp1_{id} - _gp2_{id}, _gp1_{id} - _gp2_{id}), 0.0, 1.0))",
                        ff(g.c1.r), ff(g.c1.g), ff(g.c1.b),
                        ff(g.c2.r), ff(g.c2.g), ff(g.c2.b),
                    )
                }
                Coloring::Noise(n) => {
                    self.ensure_noise();
                    let id = self.fresh_id();
                    lines.push_str(&format!(
                        "    float _nv{id} = fbm_noise(p, {});\n",
                        ff(n.seed as f32)
                    ));
                    format!(
                        "mix(vec3({}, {}, {}), vec3({}, {}, {}), _nv{id})",
                        ff(n.c1.r), ff(n.c1.g), ff(n.c1.b),
                        ff(n.c2.r), ff(n.c2.g), ff(n.c2.b),
                    )
                }
            },

            DistanceFieldEnum::Add(add) => {
                // Need distance of each branch to pick the closer one's color
                let ld = {
                    let mut dl = String::new();
                    let expr = self.emit_distance(&add.left, &mut dl);
                    lines.push_str(&dl);
                    expr
                };
                let rd = {
                    let mut dr = String::new();
                    let expr = self.emit_distance(&add.right, &mut dr);
                    lines.push_str(&dr);
                    expr
                };
                let id = self.fresh_id();
                lines.push_str(&format!("    float _ld{id} = {ld};\n"));
                lines.push_str(&format!("    float _rd{id} = {rd};\n"));

                let lc = {
                    let mut cl = String::new();
                    let expr = self.emit_color(&add.left, &mut cl);
                    lines.push_str(&cl);
                    expr
                };
                let rc = {
                    let mut cr = String::new();
                    let expr = self.emit_color(&add.right, &mut cr);
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
        assert!(shader.contains("length(p"));
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
}
