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

struct AppState {
    c: SceneNode,
    rot: UnitQuaternion<f32>,
}

struct SdfBlock{

}

#[derive(Eq, PartialEq, Hash)]
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
    block_lookup : HashMap<SdfKey, SdfBlock>
}



const sqrt_3 : f32 = 1.73205080757;
impl SdfScene {
    fn callback(&self, block : &SdfBlock, p : Vec3f, size: f32, sdf: &DistanceFieldEnum, block_size : f32 ){

    }

    fn skip_block(&self, p: Vec3, size: f32) -> bool {
        
        false
    }

    pub fn iterate_scene(&mut self, p : Vec3, size : f32 ){
        let (d, omodel) = self.sdf.distance_and_optiomize(p, size);
        if ( d > size *  sqrt_3) {
            return;
        }
        if(self.skip_block(p, size)){
           return;
        }
       
            let d2 = (p - self.eye_pos).norm();
            let mut block_size = self.block_size;
            if( d2 < 50.0){
       
            }else if(d2 < 200.0){
               block_size = self.block_size * 2.0;
            }else if(d2 < 500.0){
               block_size = self.block_size * 4.0;
            }else if(d2 < 900.0){
               block_size = self.block_size * 8.0;
            }else if(d2 < 1500.0){
               block_size = self.block_size * 16.0;
            }else{
               block_size = self.block_size * 32.0;
            }  
            if (size <= block_size) {
               
               let key = SdfKey{x : p.x as i32, y: (p.y as i32), z : p.z as i32, w : block_size as i32};
                
                let val = self.block_lookup.entry(key).or_insert(SdfBlock {  });
                //self.callback(val, p, size, &omodel, block_size);
               
               return;
            } 
       
        
            
            let s2 = size / 2.0;
            let o = [-s2,s2];
            
            let sizevec = Vec3::new(s2 * 0.5, s2 * 0.5, s2 * 0.5);
            
            for i in 0..8 
            {  
               let offset = Vec3::new(o[i & 1] as f32, o[(i >> 1) & 1] as f32, o[(i >> 2) & 1] as f32);
               let p = offset + p;
               self.iterate_scene(p, s2);
            }
           
         }
       }
       

impl AppState {
   
}

impl State for AppState {
    fn step(&mut self, _: &mut Window) {
        self.c.prepend_to_local_rotation(&self.rot)
    }


}

fn main() {

    let aabb2 = Aabb::new(Vec3f::new(2.0,0.0,0.0), Vec3f::new(1.0, 1.0, 1.5))
       .color(Rgba([255,255,255,255]));
    let grad = Gradient::new(Vec3f::new(1.9,0.0,0.0), Vec3f::new(2.1,0.0,0.0)
    , Rgba([255,0,0,255]), Rgba([255,255,255,255]), Rc::new(aabb2.into()));
    let sphere = Sphere::new(Vec3f::new(-2.0,0.0,0.0), 2.0).color(Rgba([255,0,0,255]));
    let grad2 = Gradient::new(Vec3f::new(0.0,-0.2 + 0.2,0.0), Vec3f::new(0.0,0.2+ 0.2,0.0)
    , Rgba([255,255,255,0]), Rgba([0,0,255,255]), Rc::new(sphere.into()));

    let grad3 =  Gradient::new(Vec3f::new(-2.2, 0.0, 0.0), Vec3f::new(-1.8,0.0,0.0), 
            Rgba([0,255,0,255]), 
            Rgba([0,0,255,0]), 
            Rc::new(grad2.into()));
    let noise = Noise::new(123, Rgba([95,155,55,255]), 
        Rgba([255,255,255,0]), Rc::new(grad3.into()));

    let sdf = DistanceFieldEnum::Empty{}.Insert2(noise) 
        .Insert2(grad)
        ;

    let d = sdf.distance(Vec3f::new(6.5, 5.0, 0.0));
    println!("distance: {}", d);

    //let sdf2 = sdf.optimize_bounds();

    let mut r = VertexesList::new();
    
    marching_cubes_sdf(&mut r, &sdf, Vec3f::zeros(), 5.0, 0.4);
    
    let mut window = Window::new("Kiss3d: wasm example");

    let meshtex = r.to_mesh(&sdf);
    let mut mesh = meshtex.0;
    let tex = meshtex.1;
    tex.save("test.png").unwrap();
    
    let mut c = window.add_mesh(Rc::new(RefCell::new(mesh)), Vec3f::new(0.2, 0.2, 0.2));//window.add_cube(0.5, 0.5, 0.5);
    let mut tm = TextureManager::new();

    c.append_translation(&Translation3::new(0.0, 0.0, 1.0));
    
    let mut tex2 = tm.add_image(tex, "Hello");
    
    c.set_texture(tex2);
    

    c.enable_backface_culling(true);
    window.set_light(Light::StickToCamera);
    
    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.000);
    let state = AppState { c, rot };

    window.render_loop(state);
}
