use crate::vec3::Vec3;

pub type Mat4 = [f32; 16];

pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fovy / 2.0).tan();
    let nf = near - far;
    [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) / nf, -1.0,
        0.0, 0.0, 2.0 * far * near / nf, 0.0,
    ]
}

pub fn view(eye: Vec3, dir: Vec3, world_up: Vec3) -> Mat4 {
    let right = dir.cross(world_up).normalize();
    let up = right.cross(dir);
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

pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
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

pub fn block_model_box(block_min: Vec3, block_size: Vec3) -> Mat4 {
    [
        block_size.x, 0.0, 0.0, 0.0,
        0.0, block_size.y, 0.0, 0.0,
        0.0, 0.0, block_size.z, 0.0,
        block_min.x, block_min.y, block_min.z, 1.0,
    ]
}

pub struct Frustum {
    planes: [[f32; 4]; 6],
}

impl Frustum {
    /// Extract frustum planes from a column-major view-projection matrix.
    pub fn from_vp(vp: &Mat4) -> Frustum {
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
    pub fn cull_aabb(&self, center: Vec3, half: f32) -> bool {
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

    /// Returns true if a box with per-axis half-extents is entirely outside the frustum.
    pub fn cull_aabb_extents(&self, center: Vec3, hx: f32, hy: f32, hz: f32) -> bool {
        for p in &self.planes {
            let r = hx * p[0].abs() + hy * p[1].abs() + hz * p[2].abs();
            let dist = p[0] * center.x + p[1] * center.y + p[2] * center.z + p[3];
            if dist + r < 0.0 {
                return true;
            }
        }
        false
    }
}
