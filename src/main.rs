mod mc;
mod sdf;
mod sdf_mesh;
mod triangle_raster;

use kiss3d::camera::{ArcBall, Camera};
use kiss3d::event::{Action, MouseButton, WindowEvent};


use kiss3d::resource::TextureManager;
use sdf::*;
use sdf_mesh::*;


use std::collections::HashMap;
use std::rc::Rc;

use kiss3d::light::Light;
use kiss3d::scene::SceneNode;

use image::{Rgba};

use kiss3d::nalgebra::{
    Point2, Point3, Vector2, Vector3,
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
    render_blocks: Vec<(Vec3f, f32, SdfKey, DistanceFieldEnum)>,
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
    ) {
        self.render_blocks.push((p, size, key, sdf.clone()));
    }

    fn skip_block(&self, _p: Vec3, _size: f32) -> bool {
        false
    }

    pub fn iterate_scene(&mut self, p: Vec3, size: f32) {
        self.render_blocks.clear();
        self.iterate_scene_rec(p, size)
    }

    pub fn iterate_scene_rec(&mut self, p: Vec3, size: f32) {
        let (d, omodel) = self.sdf.distance_and_optiomize(p, size);
        if d > size * sqrt_3 {
            return;
        }
        if self.skip_block(p, size) {
            return;
        }

        let d2 = (p - self.eye_pos).norm();
        let mut block_size = self.block_size;
        if d2 < 50.0 {
        } else if d2 < 200.0 {
            block_size = self.block_size * 2.0;
        } else if d2 < 500.0 {
            block_size = self.block_size * 4.0;
        } else if d2 < 900.0 {
            block_size = self.block_size * 8.0;
        } else if d2 < 1500.0 {
            block_size = self.block_size * 16.0;
        } else {
            block_size = self.block_size * 32.0;
        }
        if size <= block_size {
            let key = SdfKey {
                x: p.x as i32,
                y: (p.y as i32),
                z: p.z as i32,
                w: block_size as i32,
            };
            self.callback(key, p, size, &omodel, block_size);

            return;
        }

        let s2 = size / 2.0;
        let o = [-s2, s2];

        for i in 0..8 {
            let offset = Vec3::new(
                o[i & 1] as f32,
                o[(i >> 1) & 1] as f32,
                o[(i >> 2) & 1] as f32,
            );
            let p = offset + p;
            self.iterate_scene_rec(p, s2);
        }
    }
}

struct AppState {
    sdf_iterator: SdfScene,
    nodes: HashMap<SdfKey, (SceneNode, DistanceFieldEnum)>,
    texture_manager: TextureManager,
    cursor_pos: Vec2,
    camera: ArcBall,
}

impl AppState {
    pub fn new(sdf_iterator: SdfScene) -> AppState {
        //let cam = FirstPerson::new(Point3::new(0.0,0.0,-5.0), Point3::new(0.0, 0.0, 0.0));
        let cam = ArcBall::new(Point3::new(0.0, 0.0, -5.0), Point3::new(0.0, 0.0, 0.0));

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
                        self.sdf_iterator.sdf = self.sdf_iterator.sdf.Insert2(newobj);
                        self.nodes.clear();
                    }
                }
                WindowEvent::CursorPos(x, y, _) => {
                    self.cursor_pos = Vec2::new(x as f32, y as f32);
                }
                _ => {}
            }
        }

        let centerpos = self.camera.at().coords.map(|x| f32::floor(x / 25.0) * 25.0);

        //self.c.prepend_to_local_rotation(&self.rot);
        self.sdf_iterator.iterate_scene(centerpos, 100.0);
        for block in &self.sdf_iterator.render_blocks {
            self.nodes.entry(block.2).or_insert_with(|| {
                println!("Node at: {:?}", block);
                let pos = block.2;
                let size = pos.w as f32;

                let mut r = VertexesList::new();

                marching_cubes_sdf(&mut r, &block.3, block.0, size, 0.2);
                println!("Mc done: {:?}", r.any());
                if r.any() {
                    let meshtex = r.to_mesh(&self.sdf_iterator.sdf);

                    let name = format!("{:?}", block.2).to_string();
                    //meshtex.1.save(format!("{}-{}-{}.png", pos.x, pos.y, pos.z));

                    let tex2 = self
                        .texture_manager
                        .add_image_or_overwrite(meshtex.1, &name);

                    let mut node =
                        win.add_mesh(Rc::new(meshtex.0.into()), Vec3f::new(1.0, 1.0, 1.0));
                    node.set_texture(tex2);
                    println!("Builrt node!");
                    return (node, block.3.clone());
                } else {
                    return (win.add_group(), block.3.clone());
                }
            });
        }
    }
}

fn main() {
    let aabb2 = Sphere::new(Vec3f::new(2.0, 0.0, 0.0), 1.0).color(Rgba([255, 255, 255, 255]));
    let grad = Noise::new(
        1543,
        Rgba([255, 255, 255, 255]),
        Rgba([100, 140, 150, 255]),
        Rc::new(aabb2.into()),
    );

    let sphere = Sphere::new(Vec3f::new(-2.0, 0.0, 0.0), 2.0).color(Rgba([255, 0, 0, 255]));

    let noise = Noise::new(
        123,
        Rgba([95, 155, 55, 255]),
        Rgba([255, 255, 255, 255]),
        Rc::new(sphere.into()),
    );

    let sphere2 = Sphere::new(Vec3f::new(0.0, -200.0, 0.0), 196.0).color(Rgba([255, 0, 0, 255]));
    let noise2 = Noise::new(
        123,
        Rgba([255, 155, 55, 255]),
        Rgba([100, 100, 100, 255]),
        Rc::new(sphere2.into()),
    );

    let sdf = DistanceFieldEnum::Empty {}
        .Insert2(noise)
        .Insert2(grad)
        .Insert2(noise2);

    let sdf2 = sdf.optimize_bounds();
    let sdf_iterator = SdfScene {
        sdf: sdf2.clone(),
        eye_pos: Vec3::zeros(),
        block_size: 5.0,
        render_blocks: Vec::new(),
    };

    let mut window = Window::new("Kiss3d: wasm example");

    window.set_light(Light::StickToCamera);

    let state = AppState::new(sdf_iterator);

    window.render_loop(state);
}
