use crate::vec3::Vec3;

#[derive(Clone, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    pub fn new(normal: Vec3, d: f32) -> Plane {
        Plane { normal, d }
    }
}

#[derive(Clone, Debug)]
pub struct Polygon {
    pub vertices: Vec<Vec3>,
}

impl Polygon {
    pub fn get_vert_positions(&self) -> &Vec<Vec3> {
        &self.vertices
    }
}

#[derive(Clone, Debug)]
pub struct CsgNode {
    polygons: Vec<Polygon>,
}

impl CsgNode {
    pub fn new_cube(size: Vec3, center: Vec3) -> CsgNode {
        let hs = size * 0.5;
        let mut polygons = Vec::new();
        let faces: [(Vec3, Vec3, Vec3, Vec3); 6] = [
            (Vec3::new(-hs.x, -hs.y, -hs.z), Vec3::new(hs.x, -hs.y, -hs.z), Vec3::new(hs.x, hs.y, -hs.z), Vec3::new(-hs.x, hs.y, -hs.z)),
            (Vec3::new(-hs.x, -hs.y, hs.z), Vec3::new(hs.x, -hs.y, hs.z), Vec3::new(hs.x, hs.y, hs.z), Vec3::new(-hs.x, hs.y, hs.z)),
            (Vec3::new(-hs.x, -hs.y, -hs.z), Vec3::new(-hs.x, -hs.y, hs.z), Vec3::new(-hs.x, hs.y, hs.z), Vec3::new(-hs.x, hs.y, -hs.z)),
            (Vec3::new(hs.x, -hs.y, -hs.z), Vec3::new(hs.x, -hs.y, hs.z), Vec3::new(hs.x, hs.y, hs.z), Vec3::new(hs.x, hs.y, -hs.z)),
            (Vec3::new(-hs.x, -hs.y, -hs.z), Vec3::new(hs.x, -hs.y, -hs.z), Vec3::new(hs.x, -hs.y, hs.z), Vec3::new(-hs.x, -hs.y, hs.z)),
            (Vec3::new(-hs.x, hs.y, -hs.z), Vec3::new(hs.x, hs.y, -hs.z), Vec3::new(hs.x, hs.y, hs.z), Vec3::new(-hs.x, hs.y, hs.z)),
        ];
        for (a, b, c, d) in faces {
            polygons.push(Polygon { vertices: vec![a + center, b + center, c + center, d + center] });
        }
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

#[derive(Clone, Debug)]
pub struct Mesh2 {
    pub polygons: Vec<Polygon>,
}

impl Mesh2 {
    pub fn from_polygons(polygons: Vec<Polygon>) -> Mesh2 {
        Mesh2 { polygons }
    }

    pub fn get_polygons(&self) -> &Vec<Polygon> {
        &self.polygons
    }

    pub fn split(&self, plane: &Plane) -> (Mesh2, Mesh2) {
        let mut front = Vec::new();
        let mut back = Vec::new();
        for poly in &self.polygons {
            let mut all_front = true;
            let mut all_back = true;
            for v in &poly.vertices {
                let d = plane.normal.dot(*v) + plane.d;
                if d > 0.0 { all_back = false; }
                if d < 0.0 { all_front = false; }
            }
            if all_front {
                front.push(poly.clone());
            } else if all_back {
                back.push(poly.clone());
            } else {
                front.push(poly.clone());
            }
        }
        (Mesh2 { polygons: front }, Mesh2 { polygons: back })
    }

    pub fn to_triangles(&self) -> (Vec<Vec3>, Vec<(usize, usize, usize)>) {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        for poly in &self.polygons {
            if poly.vertices.len() >= 3 {
                let base = verts.len();
                for v in &poly.vertices {
                    verts.push(*v);
                }
                for i in 1..poly.vertices.len() - 1 {
                    indices.push((base, base + i, base + i + 1));
                }
            }
        }
        (verts, indices)
    }
}
