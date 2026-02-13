use std::collections::{HashMap, HashSet};

use crate::{sdf, vec3::Vec3};
use sdf::*;

type Vec3f = Vec3;

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy)]
pub struct SdfKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}

/// Column-major 4x4 matrix stored as [f32; 16]
pub type Mat4 = [f32; 16];

pub fn mat4_identity() -> Mat4 {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

pub fn mat4_perspective(fov_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_rad * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect, 0.0, 0.0,                    0.0,
        0.0,        f,   0.0,                    0.0,
        0.0,        0.0, (far + near) * nf,      -1.0,
        0.0,        0.0, 2.0 * far * near * nf,  0.0,
    ]
}

pub fn mat4_look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = (center - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    [
        s.x,           u.x,           -f.x,          0.0,
        s.y,           u.y,           -f.y,          0.0,
        s.z,           u.z,           -f.z,          0.0,
        -s.dot(eye),   -u.dot(eye),   f.dot(eye),    1.0,
    ]
}

pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

fn mat4_transform_point(m: &Mat4, p: Vec3) -> Option<Vec3> {
    let x = m[0] * p.x + m[4] * p.y + m[8]  * p.z + m[12];
    let y = m[1] * p.x + m[5] * p.y + m[9]  * p.z + m[13];
    let z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
    let w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    if w.abs() < 1e-10 {
        return None;
    }
    Some(Vec3::new(x / w, y / w, z / w))
}

fn is_in_frustum(cam: &Mat4, pos: Vec3, size: f32) -> bool {
    let mut min = Vec3::new(0.0, 0.0, 0.0);
    let mut max = min;
    let mut first = true;
    for i in 0..8 {
        let x = i & 1;
        let y = (i >> 1) & 1;
        let z = (i >> 2) & 1;
        let v = pos + Vec3::new(x as f32 - 0.5, y as f32 - 0.5, z as f32 - 0.5) * size;
        let v4 = mat4_transform_point(cam, v);
        if v4.is_none() {
            continue;
        }
        let v = v4.unwrap();

        if first {
            first = false;
            min = v;
            max = min;
        } else {
            min = min.min(v);
            max = max.max(v);
        }
    }

    if min.x > 1.001 || max.x < -1.001 {
        return false;
    }
    if min.y > 1.001 || max.y < -1.001 {
        return false;
    }
    if min.z > 1.001 || max.z < -1.001 {
        return false;
    }

    return true;
}

pub fn box_is_occluded(eye: Vec3, p: Vec3, size: f32, sdf: &DistanceFieldEnum) -> bool {
    let d0 = sdf.distance(eye);

    let dir = p - eye;
    let d1 = dir.length();
    if d1 < size * SQRT3 {
        return false;
    }
    let dir = dir / d1;

    let target = p - (dir * size);

    let mut it = eye + dir * d0;
    for _ in 0..128 {
        let d = sdf.distance(it) + size;
        assert!(!d.is_nan());
        if d > 1000.0 {
            return false;
        }
        if (it - target).dot(dir) > 0.01 {
            return false;
        }

        if (it - target).length() < size * SQRT3 {
            return false;
        }
        if d < size * 0.05 {
            return true;
        }
        it = it + dir * d;
    }


    return false;
}

pub struct SdfScene {
    pub sdf: DistanceFieldEnum,
    pub eye_pos: Vec3f,
    pub block_size: f32,
    pub render_blocks: Vec<(Vec3f, f32, SdfKey, DistanceFieldEnum, f32)>,
    pub cam: Mat4,
    cache: HashSet<DistanceFieldEnum>,
    map: HashMap<SdfKey, DistanceFieldEnum>,
}

const SQRT3: f32 = 1.73205080757;
impl SdfScene {
    pub fn new(sdf: DistanceFieldEnum) -> SdfScene {
        SdfScene {
            sdf,
            eye_pos: Vec3::zeros(),
            block_size: 2.0,
            render_blocks: Vec::new(),
            cam: mat4_look_at(
                Vec3::new(0.0, 0.0, -5.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            cache: HashSet::new(),
            map: HashMap::new(),
        }
    }

    pub fn with_eye_pos(mut self, v: Vec3f) -> SdfScene {
        self.eye_pos = v;
        return self;
    }

    fn callback(
        &mut self,
        key: SdfKey,
        p: Vec3f,
        size: f32,
        sdf: &DistanceFieldEnum,
        _block_size: f32,
        scale: f32,
    ) {
        self.render_blocks.push((p, size, key, sdf.clone(), scale));
    }

    fn skip_block(&self, p: Vec3, size: f32) -> bool {
        return self.skip_block2(p, size);
    }

    fn skip_block2(&self, p: Vec3, size: f32) -> bool {
        if !is_in_frustum(&self.cam, p, size * 2.0) {
            return true;
        }

        let eye = self.eye_pos;
        let occluded = box_is_occluded(eye, p, size, &self.sdf);
        if occluded {
            return true;
        }

        return false;
    }

    pub fn iterate_scene(&mut self, p: Vec3, size: f32) {
        self.render_blocks.clear();
        self.iterate_scene_rec(p, size, true)
    }

    fn iterate_scene_rec(&mut self, cell_position: Vec3, cell_size: f32, update: bool) {
        let key = SdfKey {
            x: cell_position.x as i32,
            y: cell_position.y as i32,
            z: cell_position.z as i32,
            w: cell_size as i32,
        };

        let key_exists = self.map.contains_key(&key);
        let mut update2 = !key_exists;
        let (d, _omodel) = if !update && key_exists {
            let map = self.map[&key].clone();
            (map.distance(cell_position), map)
        } else {
            let r = self
                .sdf
                .distance_and_optimize(cell_position, cell_size, &mut self.cache);
            if update && key_exists {
                let current_map = self.map.get(&key).unwrap();
                if false == current_map.eq(&r.1) {
                    update2 = true;
                    self.map.insert(key, r.1.clone());
                    println!("updated map at: {:?}", key);
                }
            } else {
                self.map.insert(key, r.1.clone());
                println!("updated map at: {:?}", key);
            }
            r
        };
        if d > cell_size * SQRT3  {
            return;
        }

        if d < -cell_size * SQRT3 {
            return;
        }

        if cell_size < 0.5 {
            return;
        }

        if self.skip_block(cell_position, cell_size) {
            return;
        }

        let cell_distance = (cell_position - self.eye_pos).length();

        let lod_level = (0.05 * cell_distance / self.block_size)
            .log2()
            .floor()
            .max(0.0);

        let lod_cell_size = self.block_size * 2.0_f32.powf(lod_level);

        if cell_size <= lod_cell_size {
            let omodel2 = self
                .sdf
                .distance_and_optimize(cell_position, cell_size, &mut self.cache)
                .1;
            self.callback(
                key,
                cell_position,
                cell_size,
                &omodel2,
                cell_size,
                lod_level,
            );

            return;
        }

        let s2 = cell_size / 2.0;
        let o = [-s2, s2];

        for i in 0..8 {
            let offset = Vec3::new(
                o[i & 1] as f32,
                o[(i >> 1) & 1] as f32,
                o[(i >> 2) & 1] as f32,
            );
            let p = offset + cell_position;
            self.iterate_scene_rec(p, s2, update2);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_optimize_bounds() {
        let sdf = build_test();
        let sdf2 = sdf.optimize_bounds();

        let mut sdf_iterator = SdfScene::new(sdf2);

        sdf_iterator.iterate_scene(Vec3::new(0.0, 0.0, 0.0), 2.0 * 512.0);
        println!("Count: {}", sdf_iterator.render_blocks.len());
        for _block in sdf_iterator.render_blocks {
            //println!("{:?}", block.2);
        }
    }

    #[test]
    fn test_progressive_holes() {
        let mut sdf: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 100.0).into();
        let mut prev = sdf.clone();
        for i in 0..10 {
            let start = Vec3::new(0.0, 0.0, 110.0);
            let dir = Vec3::new(0.0, 0.0, -1.0);
            let r = sdf.cast_ray(start, dir, 1000.0).unwrap();
            println!("{} {}", r.0, r.1);
            let newobj = Sphere::new(r.1, 2.0);
            let sub = Subtract::new(sdf.clone(), newobj, 0.5);
            sdf = sub.into();
            let r = sdf.optimized_for_block(Vec3::new(0.0, 0.0, 95.0), 2.0, &mut HashSet::new());
            sdf = sdf.optimize_bounds();
            println!("{}", r);
            if i > 30 {
                assert!(prev.eq(&r));
            }
            prev = r.as_ref().clone();
        }
    }
    #[test]
    fn test_occlusion() {
        let s1: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 100.0).into();
        let s2: DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 120.0), 10.0).into();
        let a1: DistanceFieldEnum = Add::new(s1, s2).into();

        let occluded = box_is_occluded(
            Vec3::new(0.0, 0.0, -110.0),
            Vec3::new(0.0, 0.0, 120.0),
            10.0,
            &a1,
        );
        assert!(occluded);
        let occluded2 = box_is_occluded(
            Vec3::new(0.0, 0.0, 110.0),
            Vec3::new(0.0, 0.0, 120.0),
            10.0,
            &a1,
        );
        assert!(!occluded2);

        let occluded3 = box_is_occluded(
            Vec3::new(10.0, 10.0, -110.0),
            Vec3::new(0.0, 0.0, 120.0),
            10.0,
            &a1,
        );
        assert!(occluded3);
        let occluded4 = box_is_occluded(
            Vec3::new(0.0, 0.0, -95.0),
            Vec3::new(0.0, 0.0, 120.0),
            10.0,
            &a1,
        );
        assert!(occluded4);
        let occluded5 = box_is_occluded(
            Vec3::new(0.0, 0.0, 95.0),
            Vec3::new(0.0, 0.0, 120.0),
            10.0,
            &a1,
        );
        assert!(occluded5);
        let occluded5 = box_is_occluded(
            Vec3::new(0.0, 95.0, -50.0),
            Vec3::new(0.0, 0.0, 120.0),
            10.0,
            &a1,
        );
        assert!(occluded5);
        let occluded5 = box_is_occluded(
            Vec3::new(0.0, 95.0, 50.0),
            Vec3::new(0.0, 0.0, 120.0),
            10.0,
            &a1,
        );
        assert!(!occluded5);
        let occluded5 = box_is_occluded(
            Vec3::new(0.0, 100.0, -45.0),
            Vec3::new(0.0, 0.0, 120.0),
            10.0,
            &a1,
        );
        assert!(occluded5);
        let occluded5 = box_is_occluded(
            Vec3::new(0.0, 100.0, 45.0),
            Vec3::new(0.0, 0.0, 120.0),
            10.0,
            &a1,
        );
        assert!(!occluded5);
    }
}
