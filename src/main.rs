mod mc;
mod sdf;
mod sdf_mesh;
mod triangle_raster;

use noise::{Perlin, NoiseFn};

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::rc::Rc;
use std::collections::HashMap;
use kiss3d::resource::TextureManager;
use sdf::*;
use sdf_mesh::*;

use kiss3d::light::Light;
use kiss3d::scene::SceneNode;

use kiss3d::window::{State, Window};
use kiss3d::nalgebra::{UnitQuaternion, Vector3, Translation3, Vector2};
use image::{Rgba, ImageBuffer, RgbaImage, DynamicImage};

use crate::triangle_raster::Triangle;

type Vec3f = Vector3<f32>;
type Vec3 = Vec3f;
type Vec2 = Vector2<f32>;


#[derive(Clone)]
struct SdfBlock{

}

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy)]
struct SdfKey{
    x : i32,
    y : i32,
    z : i32,
    w : i32
}


struct SdfScene {
    sdf : DistanceFieldEnum,
    eye_pos: Vec3f,
    block_size: f32,
    render_blocks : Vec<(Vec3f, f32, SdfKey)>
}



const sqrt_3 : f32 = 1.73205080757;
impl SdfScene {
    fn callback(&mut self, key : SdfKey, p : Vec3f, size: f32, sdf: &DistanceFieldEnum, block_size : f32 ){
        self.render_blocks.push((p, size, key));
    }

    fn skip_block(&self, p: Vec3, size: f32) -> bool {
        
        false
    }

    pub fn iterate_scene(&mut self, p : Vec3, size : f32 ){
        self.render_blocks.clear();
        self.iterate_scene_rec(p, size)
    }

    pub fn iterate_scene_rec(&mut self, p : Vec3, size : f32 ){
        println!("{:?}", p);
        let (d, omodel) = self.sdf.distance_and_optiomize(p, size);
        if  d > size * sqrt_3 {
            return;
        }
        if self.skip_block(p, size) {
           return;
        }
       
            let d2 = (p - self.eye_pos).norm();
            let mut block_size = self.block_size;
            if d2 < 50.0{
       
            }else if d2 < 200.0 {
               block_size = self.block_size * 2.0;
            }else if d2 < 500.0{
               block_size = self.block_size * 4.0;
            }else if d2 < 900.0{
               block_size = self.block_size * 8.0;
            }else if d2 < 1500.0{
               block_size = self.block_size * 16.0;
            }else{
               block_size = self.block_size * 32.0;
            }  
            if size <= block_size {
               
               let key = SdfKey{x : p.x as i32, y: (p.y as i32), z : p.z as i32, w : block_size as i32};
               self.callback(key, p, size, &omodel, block_size);
                
               return;
            } 
       
        
            
            let s2 = size / 2.0;
            let o = [-s2,s2];
            
            for i in 0..8 
            {  
               let offset = Vec3::new(o[i & 1] as f32, o[(i >> 1) & 1] as f32, o[(i >> 2) & 1] as f32);
               let p = offset + p;
               self.iterate_scene_rec(p, s2);
            }
           
         }
       }
       
struct AppState {
        sdf_iterator : SdfScene,
        nodes: HashMap<SdfKey, SceneNode>,
        texture_manager : TextureManager
}
    

impl AppState {
   pub fn new(sdf_iterator : SdfScene ) -> AppState{
    AppState {  sdf_iterator, nodes: HashMap::new(), texture_manager : TextureManager::new() }
   }
}

impl State for AppState {
    fn step(&mut self, win: &mut Window) {
        //self.c.prepend_to_local_rotation(&self.rot);
        self.sdf_iterator.iterate_scene(self.sdf_iterator.eye_pos, 5.0);
        for block in &self.sdf_iterator.render_blocks {
            self.nodes.entry(block.2).or_insert_with(||{
                println!("Node at: {:?}", block);
                let pos = block.2;
                let size = pos.w as f32;

                let mut r = VertexesList::new();
                
                marching_cubes_sdf(&mut r, &self.sdf_iterator.sdf, block.0,  size, 0.4);
                println!("Mc done: {:?}", r.any());
                if r.any() {
                let meshtex = r.to_mesh(&self.sdf_iterator.sdf);  

                let name = format!("{:?}", block.2).to_string();
                meshtex.1.save(format!("{}-{}-{}.png", pos.x, pos.y, pos.z));

                let mut tex2 = self.texture_manager.add_image(meshtex.1, &name);
  
                
                let mut node = win.add_mesh(Rc::new(meshtex.0.into()), Vec3f::new(1.0, 1.0, 1.0));
                node.set_texture(tex2);
                println!("Builrt node!");
                return node;
                }else{
                    return win.add_group();
                }
            });
        }
    }
}

fn main() {

    let aabb2 = Sphere::new(Vec3f::new(2.0,0.0,0.0), 1.0)
       .color(Rgba([255,255,255,255]));
    let grad = Noise::new( 1543, Rgba([255,255,255,255]), Rgba([100,140,150,255]), Rc::new(aabb2.into()));
    
    let sphere = Sphere::new(Vec3f::new(-2.0,0.0,0.0), 2.0).color(Rgba([255,0,0,255]));
    
    let noise = Noise::new(123, Rgba([95,155,55,255]), 
        Rgba([255,255,255,255]), Rc::new(sphere.into()));

    let sphere2 = Sphere::new(Vec3f::new(0.0,-1000.0,0.0), 995.0).color(Rgba([255,0,0,255]));
    let noise2 = Noise::new(123, Rgba([255,155,55,255]), 
        Rgba([100,100,100,255]), Rc::new(sphere2.into()));


    let sdf = DistanceFieldEnum::Empty{}.Insert2(noise) 
        .Insert2(grad)
        .Insert2(noise2)
        ;

    let sdf2 = sdf.optimize_bounds();
    let sdf_iterator = SdfScene { sdf: sdf2.clone(), eye_pos: Vec3::zeros(),
        block_size: 5.0, render_blocks: Vec::new() };

    let mut window = Window::new("Kiss3d: wasm example");

    window.set_light(Light::StickToCamera);
    
    let state = AppState::new(sdf_iterator);
    
    
    window.render_loop(state);
}
