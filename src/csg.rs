use crate::vec3::Vec3;

#[derive(Clone, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub w: f32,
}

impl Plane {
    pub fn new(normal: Vec3, w: f32) -> Plane {
        Plane { normal, w }
    }
}

#[derive(Clone, Debug)]
pub struct Vertex {
    pub pos: Vec3,
}

#[derive(Clone, Debug)]
pub struct Polygon {
    pub vertices: Vec<Vertex>,
}

impl Polygon {
    pub fn get_vert_positions(&self) -> Vec<Vec3> {
        self.vertices.iter().map(|v| v.pos).collect()
    }
}

#[derive(Clone, Debug)]
pub struct CsgNode {
    pub polygons: Vec<Polygon>,
}

impl CsgNode {
    pub fn new_cube(size: Vec3, center: Vec3) -> CsgNode {
        let s = size * 0.5;
        let faces: Vec<([Vec3; 4], Vec3)> = vec![
            ([Vec3::new(-1.0,-1.0,-1.0), Vec3::new(-1.0,-1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0,-1.0)], Vec3::new(-1.0, 0.0, 0.0)),
            ([Vec3::new( 1.0,-1.0,-1.0), Vec3::new( 1.0, 1.0,-1.0), Vec3::new( 1.0, 1.0, 1.0), Vec3::new( 1.0,-1.0, 1.0)], Vec3::new( 1.0, 0.0, 0.0)),
            ([Vec3::new(-1.0,-1.0,-1.0), Vec3::new( 1.0,-1.0,-1.0), Vec3::new( 1.0,-1.0, 1.0), Vec3::new(-1.0,-1.0, 1.0)], Vec3::new( 0.0,-1.0, 0.0)),
            ([Vec3::new(-1.0, 1.0,-1.0), Vec3::new(-1.0, 1.0, 1.0), Vec3::new( 1.0, 1.0, 1.0), Vec3::new( 1.0, 1.0,-1.0)], Vec3::new( 0.0, 1.0, 0.0)),
            ([Vec3::new(-1.0,-1.0,-1.0), Vec3::new(-1.0, 1.0,-1.0), Vec3::new( 1.0, 1.0,-1.0), Vec3::new( 1.0,-1.0,-1.0)], Vec3::new( 0.0, 0.0,-1.0)),
            ([Vec3::new(-1.0,-1.0, 1.0), Vec3::new( 1.0,-1.0, 1.0), Vec3::new( 1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0)], Vec3::new( 0.0, 0.0, 1.0)),
        ];
        let polygons = faces.iter().map(|(verts, _)| {
            Polygon {
                vertices: verts.iter().map(|v| {
                    Vertex { pos: center + *v * s }
                }).collect(),
            }
        }).collect();
        CsgNode { polygons }
    }

    pub fn join(&self, other: &CsgNode) -> CsgNode {
        let mut polygons = self.polygons.clone();
        polygons.extend(other.polygons.clone());
        CsgNode { polygons }
    }

    pub fn all_polygons(&self) -> Vec<Polygon> {
        self.polygons.clone()
    }
}

pub struct Mesh2 {
    polygons: Vec<Polygon>,
}

impl Mesh2 {
    pub fn from_polygons(polygons: Vec<Polygon>) -> Mesh2 {
        Mesh2 { polygons }
    }

    pub fn split(&self, plane: &Plane) -> (Mesh2, Mesh2) {
        let mut front = Vec::new();
        let mut back = Vec::new();
        for poly in &self.polygons {
            let mut all_front = true;
            let mut all_back = true;
            for v in &poly.vertices {
                let d = v.pos.dot(plane.normal) - plane.w;
                if d > 0.001 { all_back = false; }
                if d < -0.001 { all_front = false; }
            }
            if all_front {
                front.push(poly.clone());
            } else if all_back {
                back.push(poly.clone());
            } else {
                front.push(poly.clone());
                back.push(poly.clone());
            }
        }
        (Mesh2 { polygons: front }, Mesh2 { polygons: back })
    }

    pub fn get_polygons(&self) -> &[Polygon] {
        &self.polygons
    }

    pub fn to_triangles(&self) -> (Vec<Vec3>, Vec<(usize, usize, usize)>) {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        for poly in &self.polygons {
            if poly.vertices.len() < 3 { continue; }
            let base = verts.len();
            for v in &poly.vertices {
                verts.push(v.pos);
            }
            for i in 1..poly.vertices.len()-1 {
                indices.push((base, base + i, base + i + 1));
            }
        }
        (verts, indices)
    }
}
