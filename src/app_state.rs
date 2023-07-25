use std::{collections::HashMap, rc::Rc};

use crate::{sdf, sdf_scene::{SdfScene, SdfKey}, sdf_mesh::{VertexesList, marching_cubes_sdf}, vec3::Vec3};
use image::Rgba;
use sdf::*;

use kiss3d::{nalgebra::{Vector3, Point3, Point2, Vector2, Translation3}, resource::TextureManager, camera::{ArcBall, Camera, FirstPerson}, scene::SceneNode, window::{State, Window}, event::{MouseButton, WindowEvent, Action}};

type Vec3f = Vec3;
type Vec2 = Vector2<f32>;

pub struct AppState {
    sdf_iterator: SdfScene,
    nodes: HashMap<SdfKey, (SceneNode, DistanceFieldEnum, f32, Vec3)>,
    texture_manager: TextureManager,
    cursor_pos: Vec2,
    camera: FirstPerson,
}

impl AppState {
    pub fn new(sdf_iterator: SdfScene) -> AppState {
        let cam = FirstPerson::new(Point3::new(0.0,0.0,-5.0), Point3::new(0.0, 0.0, 0.0));
        /*let mut cam = ArcBall::new_with_frustrum(1.0, 0.1, 1000.0, 
            Point3::new(0.0, 0.0, -5.0), 
            Point3::new(0.0, 0.0, 0.0));
        */
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
                            .cast_ray(Vec3::new(at.x, at.y, at.z).into(), unp.1.into(), 1000.0);

                    if let Some((_, p)) = col {
                        let newobj = Sphere::new(p, 2.0).color(Rgba([255, 0, 0, 255]));
                        
                        let sub = Subtract::new(self.sdf_iterator.sdf.clone(), newobj, 0.5);
                        self.sdf_iterator.sdf = sub.into();
                        println!("old: {}", self.sdf_iterator.sdf);
                        self.sdf_iterator.sdf = self.sdf_iterator.sdf.optimize_bounds();
                        println!("new: {}", self.sdf_iterator.sdf);
                        
                    }
                }
                WindowEvent::CursorPos(x, y, _) => {
                    self.cursor_pos = Vec2::new(x as f32, y as f32);
                }
                _ => {}
            }
        }

        let centerpos : Vec3 = self.camera.eye()
            .coords.xyz().map(|x| f32::floor(x / 16.0) * 16.0).into();
        self.sdf_iterator.eye_pos = self.camera.eye().to_homogeneous().xyz().into();
        self.sdf_iterator.cam = (&self.camera).into();
        self.sdf_iterator.iterate_scene(centerpos, 256.0);

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
                let newsdf = block.3.optimized_for_block(block.0.into(), size);
                marching_cubes_sdf(&mut r, &newsdf, block.0.into(), size, 0.2 * 2.0_f32.powf(block.4));
                
                if r.any() {
                    if false {
                        println!("Cube: {} {}", size, pos);
                
                        let mut cube = win.add_cube(1.0, 1.0, 1.0);
                    
                        cube.set_local_scale(size * 1.99, size * 1.99, size* 1.99);
                    
                        cube.set_local_translation(Translation3::new(pos.x as f32, pos.y as f32, pos.z as f32));
                    
                        return (cube, block.3.clone(), size, pos);
                    }
    
                    let meshtex = r.to_mesh(&newsdf);

                    let name = format!("{:?}", block.2).to_string();

                    let tex2 = self
                        .texture_manager
                        .add_image_or_overwrite(meshtex.1, &name);

                    let mut node =
                        win.add_mesh(Rc::new(meshtex.0.into()), Vec3::new(1.0, 1.0, 1.0).into());
                    node.set_texture(tex2);
                    return (node, block.3.clone(), size, pos);
                } else {
                    return (win.add_group(), block.3.clone(), size, pos);
                }
            });
            
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
