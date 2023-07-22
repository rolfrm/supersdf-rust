use crate::{sdf, vec3::Vec3};
use sdf::*;

use kiss3d::nalgebra::{Vector2};

type Vec3f = Vec3;
type Vec2 = Vector2<f32>;

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy)]
pub struct SdfKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}

pub struct SdfScene {
    pub  sdf: DistanceFieldEnum,
    pub eye_pos: Vec3f,
    pub block_size: f32,
    pub render_blocks: Vec<(Vec3f, f32, SdfKey, DistanceFieldEnum, f32)>,
}

const SQRT3: f32 = 1.73205080757;
impl SdfScene {

    pub fn new(sdf : DistanceFieldEnum) -> SdfScene {
        SdfScene { sdf: sdf, eye_pos: Vec3::zeros(), block_size: 2.0, render_blocks: Vec::new() }
    } 

    fn callback(
        &mut self,
        key: SdfKey,
        p: Vec3f,
        size: f32,
        sdf: &DistanceFieldEnum,
        _block_size: f32,
        scale: f32
    ) {
        self.render_blocks.push((p, size, key, sdf.clone(), scale));
    }

    fn skip_block(&self, _p: Vec3, _size: f32) -> bool {
        false
    }

    pub fn iterate_scene(&mut self, p: Vec3, size: f32) {
        self.render_blocks.clear();
        self.iterate_scene_rec(p, size)
    }

    // The `iterate_scene_rec` function is a recursive function that works by dividing the scene into cells, and for each cell, it checks 
    // if it's close enough to the scene, or if it's too small or not, and if so, it skips the current cell. If the cell is relevant, it will
    // divide it into eight smaller cells and repeat the process.
    fn iterate_scene_rec(&mut self, cell_position: Vec3, cell_size: f32) {
    
        // Calculate the SDF distance and check if optimizations can be made for the sdf in a local scope.
        //let (d, omodel) = (self.sdf.distance(cell_position), self.sdf.clone());
        let (d, omodel) = self.sdf.distance_and_optimize(cell_position, cell_size);
        if d > cell_size * SQRT3 {
            return;
        }
        if cell_size < 0.9 {
            return;
        }

        // future frustum culling.
        if self.skip_block(cell_position, cell_size) {
            return;
        }

        // Calculate the distance of the cell from the eye position
        let cell_distance = (cell_position - self.eye_pos).length();
        
        // Calculate the LOD level based on the distance of the cell.
        let lod_level = (cell_distance * 0.5 / self.block_size).log2().floor().max(0.0) * 0.5;
        
        // Calculate the cell size. Farther away -> bigger blocks with lower resolution.
        let lod_cell_size = self.block_size * 2.0_f32.powf(lod_level);

        // if the size of the current cell is less than the lod cell size
        if cell_size <= lod_cell_size {
            let key = SdfKey {
                x: cell_position.x as i32,
                y: cell_position.y as i32,
                z: cell_position.z as i32,
                w: cell_size as i32,
            };

            self.callback(key, cell_position, cell_size, &omodel, cell_size, lod_level);
            
            return;
        }

        // If the cell is not returned early, then we divide the cell into eight smaller cells and repeat the process for each.
        let s2 = cell_size / 2.0;
        let o = [-s2, s2];

        for i in 0..8 {
            let offset = Vec3::new(
                o[i & 1] as f32,
                o[(i >> 1) & 1] as f32,
                o[(i >> 2) & 1] as f32,
            );
            let p = offset + cell_position;
            self.iterate_scene_rec(p, s2);
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
    
        let mut sdf_iterator = SdfScene {
            sdf: sdf2,
            eye_pos: Vec3::zeros(),
            block_size: 2.0,
            render_blocks: Vec::new(),
        };

        sdf_iterator.iterate_scene(Vec3::new(0.0, 0.0, 0.0), 64.0);
        for block in sdf_iterator.render_blocks {
            println!("{:?}", block.2);
            
        }
    }



}
