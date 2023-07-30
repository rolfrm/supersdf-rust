use std::collections::{HashSet, HashMap};

use crate::{sdf, vec3::Vec3};
use sdf::*;

use kiss3d::{nalgebra::{Vector2, Point3, Matrix4}, camera::{Camera, ArcBall, FirstPerson}};

type Vec3f = Vec3;
type Vec2 = Vector2<f32>;

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy)]
pub struct SdfKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}

fn is_in_frustum(cam: &Matrix4<f32>, pos : Vec3, size : f32) -> bool{
 
    let mut min = Vec3::new(0.0, 0.0, 0.0);
    let mut max = min;
    let mut first = true;
    for i in 0..8 {
       let x = i & 1;
       let y = (i >> 1) & 1;
       let z = (i >> 2) & 1;
       let v = pos + Vec3::new(x as f32 - 0.5,y as f32 - 0.5,z as f32 - 0.5) * size;
       let v2 : Point3<f32> = v.into();
       let v3 = (cam * v2.to_homogeneous());
       let v4 = Point3::from_homogeneous(v3);
       if let None = v4 {
        println!("Issue: {:?}", v3);
          continue;
       }
       let v: Vec3 = v4.unwrap().into();

       if first {
          first = false;
          min = v.into();
          max = min;
       }else{
          min = min.min(v.into());
          max = max.max(v.into());
       }
    }
  
    if min.x > 1.0 || max.x < -1.0 {
         return false;
    }
    if min.y > 1.0 || max.y < -1.0 {
         return false;
    }
    if min.z > 1.0 || max.z < -1.0 {
       return false;
    }
  
    return true;
  }

pub enum CameraEnum{
    ArcBall(ArcBall),
    FirstPerson(FirstPerson),
    None
}

impl Into<CameraEnum> for &FirstPerson {
    fn into(self) -> CameraEnum {
        CameraEnum::FirstPerson(self.clone())
    }
}

impl Into<CameraEnum> for &ArcBall {
    fn into(self) -> CameraEnum {
        CameraEnum::ArcBall(self.clone())
    }
}

pub struct SdfScene {
    pub  sdf: DistanceFieldEnum,
    pub eye_pos: Vec3f,
    pub block_size: f32,
    pub render_blocks: Vec<(Vec3f, f32, SdfKey, DistanceFieldEnum, f32)>,
    pub cam: Matrix4<f32>,
    cache: HashSet<DistanceFieldEnum>,
    map : HashMap<SdfKey, DistanceFieldEnum>
}

const SQRT3: f32 = 1.73205080757;
impl SdfScene {

    pub fn new(sdf : DistanceFieldEnum) -> SdfScene {
        SdfScene { sdf: sdf, eye_pos: Vec3::zeros(), 
            block_size: 2.0, 
            render_blocks: Vec::new(),
            cam : FirstPerson::new(Point3::new(0.0, 0.0, -5.0), Point3::new(0.0, 0.0, 0.0)).transformation(),
            cache: HashSet::new(),
            map: HashMap::new()
         }
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

    fn skip_block(&self, p: Vec3, size: f32) -> bool {
        !is_in_frustum(&self.cam, p, size * 2.0)
    }

    pub fn iterate_scene(&mut self, p: Vec3, size: f32) {
        self.render_blocks.clear();
        self.iterate_scene_rec(p, size, true)
    }

    // The `iterate_scene_rec` function is a recursive function that works by dividing the scene into cells, and for each cell, it checks 
    // if it's close enough to the scene, or if it's too small or not, and if so, it skips the current cell. If the cell is relevant, it will
    // divide it into eight smaller cells and repeat the process.
    fn iterate_scene_rec(&mut self, cell_position: Vec3, cell_size: f32, update : bool) {
    
        let key = SdfKey {
            x: cell_position.x as i32,
            y: cell_position.y as i32,
            z: cell_position.z as i32,
            w: cell_size as i32,
        };
        let mut update2 = update;
        let key_exists = self.map.contains_key(&key);
        update2 = !key_exists;
        let (d, omodel) = 
            if !update && key_exists {
                let map = self.map[&key].clone();
                (map.distance(cell_position), map)
            }else{
             // Calculate the SDF distance and check if optimizations can be made for the sdf in a local scope.
           //let (d, omodel) = (self.sdf.distance(cell_position), self.sdf.clone());
            
             let r = self.sdf.distance_and_optimize(cell_position, cell_size, &mut self.cache);
             if update && key_exists {
                let current_map = self.map.get(&key).unwrap();
                if false == current_map.eq(&r.1) {
                    update2 = true;
                    self.map.insert(key, r.1.clone());
                    println!("updated map at: {:?}", key);
                }

             }else{
                self.map.insert(key, r.1.clone());
                println!("updated map at: {:?}", key);
             }
             r
            };
        if d > cell_size * SQRT3 {
            return;
        }
        
        if d < -cell_size * SQRT3 {
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
        let lod_level = (0.05 * cell_distance / self.block_size).log2().floor().max(0.0);
        
        // Calculate the cell size. Farther away -> bigger blocks with lower resolution.
        let lod_cell_size = self.block_size * 2.0_f32.powf(lod_level);

        // if the size of the current cell is less than the lod cell size
        if cell_size <= lod_cell_size {
            

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
        for block in sdf_iterator.render_blocks {
            //println!("{:?}", block.2);
            
        }
        
        
    }



}
