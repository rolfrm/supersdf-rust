mod mc;
mod sdf;
mod sdf_mesh;
mod triangle_raster;

use kiss3d::camera::{ArcBall, Camera};
use kiss3d::event::{Action, MouseButton, WindowEvent};


use kiss3d::resource::TextureManager;
use sdf::*;
use sdf_mesh::*;


use std::collections::{HashMap, HashSet};
use std::os::unix::prelude::OsStringExt;
use std::rc::Rc;

use kiss3d::light::Light;
use kiss3d::scene::SceneNode;

use image::{Rgba};

use kiss3d::nalgebra::{
    Point2, Point3, Vector2, Vector3, Translation, Translation3,
};
use kiss3d::window::{State, Window};

type Vec3f = Vector3<f32>;
type Vec3 = Vec3f;
type Vec2 = Vector2<f32>;

#[derive(Clone)]
struct SdfBlock {}

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy)]
struct SdfKey {
    x: i32,
    y: i32,
    z: i32,
    w: i32,
}

struct SdfScene {
    sdf: DistanceFieldEnum,
    eye_pos: Vec3f,
    block_size: f32,
    render_blocks: Vec<(Vec3f, f32, SdfKey, DistanceFieldEnum, f32)>,
}

const sqrt_3: f32 = 1.73205080757;
impl SdfScene {
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
    pub fn iterate_scene_rec(&mut self, cell_position: Vec3, cell_size: f32) {
    
        // Calculate the SDF distance and check if optimizations can be made for the sdf in a local scope.
        //let (d, omodel) = (self.sdf.distance(cell_position), self.sdf.clone());
        let (d, omodel) = self.sdf.distance_and_optiomize(cell_position, cell_size);
        if d > cell_size * sqrt_3 {
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
        let cell_distance = (cell_position - self.eye_pos).norm();
        
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

struct AppState {
    sdf_iterator: SdfScene,
    nodes: HashMap<SdfKey, (SceneNode, DistanceFieldEnum, f32, Vec3)>,
    texture_manager: TextureManager,
    cursor_pos: Vec2,
    camera: ArcBall,
}

impl AppState {
    pub fn new(sdf_iterator: SdfScene) -> AppState {
        //let cam = FirstPerson::new(Point3::new(0.0,0.0,-5.0), Point3::new(0.0, 0.0, 0.0));
        let mut cam = ArcBall::new_with_frustrum(1.0, 0.1, 1000.0, Point3::new(0.0, 0.0, -5.0), Point3::new(0.0, 0.0, 0.0));
        
        AppState {
            sdf_iterator,
            nodes: HashMap::new(),
            texture_manager: TextureManager::new(),
            cursor_pos: Vec2::new(0.0, 0.0),
            camera: cam,
        }
    }
}

impl State for AppState {
    fn cameras_and_effect_and_renderer(
        &mut self,
    ) -> (
        Option<&mut dyn kiss3d::camera::Camera>,
        Option<&mut dyn kiss3d::planar_camera::PlanarCamera>,
        Option<&mut dyn kiss3d::renderer::Renderer>,
        Option<&mut dyn kiss3d::post_processing::PostProcessingEffect>,
    ) {
        return (Some(&mut self.camera), None, None, None);
    }

    fn step(&mut self, win: &mut Window) {
        for evt in win.events().iter() {
            match evt.value {
                WindowEvent::MouseButton(MouseButton::Button2, Action::Press, _) => {
                    let win_size = win.size();
                    let win_size2 = Vec2::new(win_size.x as f32, win_size.y as f32);

                    let unp = self.camera.unproject(
                        &Point2::new(self.cursor_pos.x, self.cursor_pos.y),
                        &win_size2,
                    );
                    let at = unp.0;
                    let col =
                        self.sdf_iterator
                            .sdf
                            .cast_ray(Vec3::new(at.x, at.y, at.z), unp.1, 1000.0);

                    if let Some((_, p)) = col {
                        let newobj = Sphere::new(p, 0.5).color(Rgba([255, 0, 0, 255]));
                        self.sdf_iterator.sdf = self.sdf_iterator.sdf.Insert2(newobj).optimize_bounds();
                        
                        //for node in self.nodes.iter_mut() {
                        //    let n = &mut node.1.0;    
                        //    n.unlink();
                        //}
                        
                        //self.nodes.clear();
                        
                        
                    }
                }
                WindowEvent::CursorPos(x, y, _) => {
                    self.cursor_pos = Vec2::new(x as f32, y as f32);
                }
                _ => {}
            }
        }

        let centerpos = self.camera.eye()
        .coords.map(|x| f32::floor(x / 32.0) * 32.0);
        self.sdf_iterator.eye_pos = self.camera.eye().to_homogeneous().xyz();
        
        //self.c.prepend_to_local_rotation(&self.rot);
        self.sdf_iterator.iterate_scene(centerpos, 64.0);

        for node in self.nodes.iter_mut() {
            let n = &mut node.1.0;    
            n.set_visible(false);
        }
        
        for block in &self.sdf_iterator.render_blocks {
            loop {
            let nd = self.nodes.entry(block.2).or_insert_with(|| {



                let pos = block.2;
                
                let size = pos.w as f32;
                let pos = block.0;
                
                let mut r = VertexesList::new();
                //println!("Optimize: {:?}", block.3);
                let newsdf = block.3.optimized_for_block(block.0, size);
                //println!("Optimized: {:?}", newsdf);
                
                marching_cubes_sdf(&mut r, &newsdf, block.0, size, 0.2 * 2.0_f32.powf(block.4));
                
                if r.any() {
                    println!("Cube: {} ({} {} {})", size, pos.x, pos.y, pos.z);
                
                    //let mut cube = win.add_cube(1.0, 1.0, 1.0);
                    //
                    //cube.set_local_scale(size * 1.99, size* 1.99, size* 1.99);
                    //
                    //cube.set_local_translation(Translation3::new(pos.x as f32, pos.y as f32, pos.z as f32));
                    //
                    //return (cube, block.3.clone(), size, pos);
                    
    
                    let meshtex = r.to_mesh(&newsdf);

                    let name = format!("{:?}", block.2).to_string();

                    let tex2 = self
                        .texture_manager
                        .add_image_or_overwrite(meshtex.1, &name);

                    let mut node =
                        win.add_mesh(Rc::new(meshtex.0.into()), Vec3f::new(1.0, 1.0, 1.0));
                    node.set_texture(tex2);
                    println!("Builrt node!");
                    return (node, block.3.clone(), size, pos);
                } else {
                    return (win.add_group(), block.3.clone(), size, pos);
                }
            });
            //println!("{} ({} {} {}) {}", nd.2, nd.3.x,nd.3.y,nd.3.z, nd.1.eq(&block.3));
        
            
            if(!nd.1.eq(&block.3)){ 
                // node changed since last time.
                nd.0.unlink();
                self.nodes.remove(&block.2);
            }else{

                nd.0.set_visible(true);
                break;
            }
        }
            
            
        }
    }
}


fn main() {
    let sdf : DistanceFieldEnum = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 2.0)
        .color(Rgba([255, 0, 0, 255]))
        .into();// build_test();
    let sdf = build_test();
    let sdf2 = sdf.optimize_bounds();

    let sdf_iterator = SdfScene {
        sdf: sdf2,
        eye_pos: Vec3::zeros(),
        block_size: 5.0,
        render_blocks: Vec::new(),
    };

    let mut window = Window::new("Kiss3d: wasm example");
    

    window.set_light(Light::StickToCamera);

    let state = AppState::new(sdf_iterator);

    window.render_loop(state);
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
