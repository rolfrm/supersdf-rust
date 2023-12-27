use std::{collections::{HashMap, HashSet}, rc::Rc};

use crate::{sdf, sdf_scene::{SdfScene, SdfKey, box_is_occluded}, sdf_mesh::{TriangleList, Triangles, marching_cubes_sdf}, vec3::Vec3};
use crate::csg::{Mesh2, CsgNode, Plane};
use image::Rgba;
use sdf::*;



use kiss3d::{nalgebra::{Vector3, Point3, Point2, Vector2, Unit, UnitQuaternion, Translation3}, resource::{TextureManager, Mesh}, camera::{ArcBall, Camera, FirstPerson}, scene::SceneNode, window::{State, Window}, event::{MouseButton, WindowEvent, Action, Key}};

type Vec2 = Vector2<f32>;

pub struct AppState {
    sdf_iterator: SdfScene,
    sdf_cache : HashSet<DistanceFieldEnum>,
    nodes: HashMap<SdfKey, (SceneNode, DistanceFieldEnum, f32, Vec3)>,
    texture_manager: TextureManager,
    cursor_pos: Vec2,
    camera: FirstPerson,
    test : Option<SceneNode>,
    time: f32
}

impl AppState {
    pub fn new(sdf_iterator: SdfScene) -> AppState {
        let cam = FirstPerson::new(sdf_iterator.eye_pos.into(), Point3::new(0.0, 0.0, 0.0));
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
            sdf_cache: HashSet::new(),
            test: None,
            time: 0.0
        }
    }
}

trait ToRGBAF {
    fn to_rgbaf(&self) -> Rgba<f32>;
}

impl ToRGBAF for Rgba<u8> {
    fn to_rgbaf(&self) -> Rgba<f32> {
        fn conv(x: u8) -> f32 {
            return f32::from(x) / f32::from(u8::MAX)
        }    
        Rgba(self.0.map(conv))
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
        self.time += 1.0;
        if let Some(n) = &mut self.test {
            let axis = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.1 + (0.00 * self.time).sin() * 1.0));
            let angle = self.time * 0.02;
            let rot = UnitQuaternion::from_axis_angle(&axis, angle);
            n.set_local_rotation(rot);
        }

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
                        let newobj = Sphere::new(p, 2.0);//.color(Rgba([255, 0, 0, 255]));
                        
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
                WindowEvent::Key(key,action ,modifiers ) => {
                    if key == Key::Return && action == Action::Press {
                        println!("Reloading cache!");
                        for node in self.nodes.iter_mut() {
                            node.1.0.unlink()
                        }
                        self.nodes.clear();
                    }
                    if key == Key::A && action == Action::Press {
                        let cb = CsgNode::new_cube(Vec3::new(1.0, 1.0, 1.0), Vec3::zeros());
                        let cb2 = CsgNode::new_cube(Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0,3.0,0.0));
                        let cb = cb.join(&cb2);
                        let cb2 = CsgNode::new_cube(Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0,3.0,3.0));
                        let cb = cb.join(&cb2);
                        let cube = Mesh2::from_polygons(cb.all_polygons());
                        let mut mesh = cube;
                        mesh = mesh.split(&Plane::new(Vec3::new(1.0, 1.0, 1.0), 0.0)).0;
                        mesh = mesh.split(&Plane::new(Vec3::new(-1.0, -1.0, 1.0), -1.9)).0;
                        mesh = mesh.split(&Plane::new(Vec3::new(0.0, 1.0, 0.0), -0.5)).0;
                        /*for i in 0..21 {
                            let i2 = i as f32 * 1312.0 + 0.5;
                            let p = Plane::new(Vec3::new(i2.cos(), i2.sin(), (i2 * 0.5).cos()).normalize(), -0.8);
                            mesh = mesh.split(&p).0;
                        }*/
                        
                        /*for i in [-1,-1,-1,-1,-1]{
                            let i2 = i as f32 * 1312.0 + 0.5;
                            let p = Plane::new(Vec3::new(i2.cos(), i2.sin(), (i2 * 0.5).cos()).normalize(), -0.5);
                            mesh = mesh.split(&p).0;
                        }*/
                        
                        //let p = Plane::new(Vec3::new(1.0, 1.0, 1.0).normalize(), -0.9);
                        //let p2 = Plane::new(Vec3::new(-1.0, -1.0, -1.0).normalize(), -0.9);   
                        //let part1 = cube.split(&p).0.split(&p2).0;
                        let mut coords: Vec<Point3<f32>> = Vec::new();
                        let mut faces = Vec::new();
                        let trigs = mesh.to_triangles();
                        println!("verts: {} faces: {} polygons: {}", trigs.0.len(), trigs.1.len(), mesh.get_polygons().len());
                        for vert in trigs.0 {
                            
                            coords.push(Point3::new(vert.x, vert.y, vert.z));
                        }
                        for idx in trigs.1 {
                            faces.push(Point3::new(idx.0 as u16, idx.1 as u16, idx.2 as u16))
                        }
                        
                        let m = Mesh::new(coords, faces, None, None, false);
                        let node = win.add_mesh(Rc::new(m.into()), Vec3::new(1.0, 1.0, 1.0).into());
                        self.test = Some(node);
                        for p in mesh.get_polygons() {
                            println!("polygons: {:?}", p.get_vert_positions().iter().fold(Vec3::zeros(), |acc,&y| acc + y));
                        }

                        
                    }
                }
                _ => {}
            }
        }

        let centerpos : Vec3 = self.camera.eye()
            .coords.xyz().map(|x| f32::floor(x / 16.0) * 16.0).into();
        self.sdf_iterator.eye_pos = self.camera.eye().to_homogeneous().xyz().into();
        self.sdf_iterator.cam = (&self.camera).transformation();
        self.sdf_iterator.iterate_scene(centerpos, 8.0 * 128.0 * 2.0);

        for node in self.nodes.iter_mut() {
            let n = &mut node.1.0;    
            n.set_visible(false);    
        }
        let eye = self.sdf_iterator.eye_pos;
        println!("eye: {}", eye);
        let mut reload_count = 0;
        for block in &self.sdf_iterator.render_blocks {
            loop {
            let nd = self.nodes.entry(block.2).or_insert_with(|| {
                let sdf2 = block.3.cached(&mut self.sdf_cache);
                let pos = block.2;
                
                let size = pos.w as f32;
                let pos = block.0;

                
               
                
                let mut r = TriangleList::new();
                let newsdf = sdf2.optimized_for_block(block.0.into(), size,&mut self.sdf_cache)
                    .cached(&mut self.sdf_cache).clone();
                
                
                marching_cubes_sdf(&mut r, &newsdf, block.0.into(), size, 0.4 * 2.0_f32.powf(block.4), size);

                //cubify(&mut r, &newsdf, block.0.into(), size, 0.4 * 2.0_f32.powf(block.4));
                //println!("mc: {} {}\n", r.len(), size as f32 / 0.4 * 2.0_f32.powf(block.4) );
                
                if r.any() {
                    let mut r2 = Triangles::from_triangle_list(&r);
                    for i in 0..0 {
                        let mut v = Vec::new();
                    
                        //r2.collapse_triangle(0);
                
                 let c = r2.triangle_count();
                 let mut size_avg = 0.0;
                 let mut max_size = -1000.0;
                 let mut min_size = 1000.0;
                 for i in 0..c {
                    let s = r2.triangle_size(i).unwrap();
                    if false && s < 0.000001 {
                        let t2 = r2.triangle(i).unwrap();
                        let t3 = r2.triangleidx(i).unwrap();
                        
                        println!("::{}    {} {} {}  ({:?})", s, t2[0], t2[1], t2[2], t3);
                        v.push(i);
                        break;
                    }
                    max_size = f32::max(max_size, s);
                    min_size = f32::min(min_size, s);
                    size_avg += s;
                 }

                 size_avg = size_avg / (c as f32);
                 if v.len() == 0 {
                    for i in 0..c {
                        let s = r2.triangle_size(i).unwrap();
                        if s < size_avg * 0.2 {
                            let t2 = r2.triangle(i).unwrap();
                            let t3 = r2.triangleidx(i).unwrap();
                            println!("----------");
                            println!("?? ({:?})", t3);
                            println!("::{} / {}   ({} {} {}) ", s, size_avg, t2[0], t2[1], t2[2]);

                            if(s < 0.00001 ){
                                
                                for x in r2.vertices {
                                    println!("--- {}", x);
                                }   
                                panic!("Triangle {} invalid", i);
    
                                //println!("{:?}", r2.triangles);
                            }
                            v.push(i);
                            break;
                        }
                     }
                 }
                 if v.len() == 0 {
                    break;
                 }

                 for i in v.iter().rev() {
                    println!("removeing triangle: {} ", *i);
                    //r2.collapse_triangle(*i)
                 }


                println!(" sizes: avg {}, min {}, max {}", size_avg, min_size, max_size);
                }
                 //r2.collapse_triangle(0);
                let r0 = r.len();
                //r = r2.to_triangle_list();
                if(!r.any()){
                    return (win.add_group(), block.3.clone(), size, pos);
                }
                println!("prev: {}, after: {} ", r0, r.len());
                
                    
                    let mut meshtex = r.to_mesh(&newsdf);

                    let name = format!("{:?}", block.2).to_string();

                    let tex2 = self
                        .texture_manager
                        .add_image_or_overwrite(meshtex.1, &name);

                    let scale = block.1 * 2.0;
                    let pos = block.0;

                    //let mut node = win.add_cube(scale * 0.5, scale * 0.5, scale * 0.5);
                    //node.set_local_translation(Translation3::new(pos.x - scale * 0.5, pos.y - scale * 0.5, pos.z - scale * 0.5));
                    //let color = block.3.color(pos);
                    
                    //node.set_color(color.r,color.g, color.b);
                    //println!("color: {} {}", color, color.a);
                    //if(color.a < 0.1){
                        //println!("invisible!");
                    //    win.remove_node(&mut node);
                    //    node.set_lines_width(3.1);
                    //}else{
                    //    node.set_lines_width(1.0);
                    
                    //}
                    let mut node =
                        win.add_mesh(Rc::new(meshtex.0.into()), Vec3::new(1.0, 1.0, 1.0).into());
                    
                    node.set_lines_width(1.0);
                    node.set_surface_rendering_activation(false);
                        
                    node.set_texture(tex2);
                    return (node, block.3.clone(), size, pos);
                } else {
                    return (win.add_group(), block.3.clone(), size, pos);
                }
            });
            
            if !nd.1.eq(&block.3){ 
                reload_count += 1;
                println!("Reload: {} ({} {} {} {})", reload_count, block.2.x, block.2.y, block.2.z, block.2.w);
                // node changed since last time.
                println!("Prev: {}", block.3);
                println!("After: {}", nd.1);
                nd.0.unlink();
                let key = block.2;
                self.nodes.remove(&key);
                
                
            }else{
                   
                nd.0.set_visible(true);
                break;
            }
        }
            
            
        }
    }
}
