use std::cmp::Ordering;

use crate::vec2::Vec2;

pub struct Line {
    from: Vec2,
    to: Vec2,
}

impl Line {
    pub fn new(from: Vec2, to: Vec2) -> Line {
        Line { from, to }
    }
    pub fn into_iter(&self) -> LineIterator {
        let mut it = LineIterator {
            from: self.from,
            to: self.to,
            d: (self.to - self.from).abs(),
            s: Vec2::new(
                match self.to.x > self.from.x {
                    true => 1.0,
                    false => -1.0,
                },
                match self.to.y > self.from.y {
                    true => 1.0,
                    false => -1.0,
                },
            ),
            err: 0.0,
        };
        it.err = it.d.x - it.d.y;
        return it;
    }

    pub fn into_iter_y(&self) -> LineYIterator {
        LineYIterator {
            line: self.into_iter(),
            last_y: None,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct LineIterator {
    from: Vec2,
    to: Vec2,
    d: Vec2,
    s: Vec2,
    err: f32,
}

impl Iterator for LineIterator {
    type Item = Vec2;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.from;
        if (self.from - self.to).dot(self.d) >= 0.0 {
            return None;
        }

        let err2 = self.err * 2.0;
        if err2 > -self.d.y {
            self.err -= self.d.y;
            self.from.x += self.s.x;
        }

        if err2 < self.d.x {
            self.err += self.d.x;
            self.from.y += self.s.y;
        }

        return Some(start);
    }
}
#[derive(Debug, Copy, Clone)]
pub struct LineYIterator {
    line: LineIterator,
    last_y: Option<f32>,
}

impl Iterator for LineYIterator {
    type Item = Vec2;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v0) = self.last_y {
            loop {
                let val = self.line.next();
                if let Some(v2) = val {
                    if v2.y != v0 {
                        self.last_y = Some(v2.y);
                        return Some(v2);
                    }
                    continue;
                } else {
                    return None;
                }
            }
        } else {
            let val = self.line.next();
            if let Some(v2) = val {
                self.last_y = Some(v2.y);
                return Some(v2);
            } else {
                return None;
            }
        }
    }
}

struct F32Range {
    now: f32,
    end: f32,
    step: f32,
}

impl F32Range {
    pub fn new(start: f32, stop: f32, step: Option<f32>) -> F32Range {
        F32Range {
            now: start,
            end: stop,
            step: match step {
                Some(s) => s,
                None => (stop - start).signum(),
            },
        }
    }

    pub fn with_step(&self, step: f32) -> F32Range {
        F32Range {
            now: self.now,
            end: self.end,
            step: step,
        }
    }
}

impl Iterator for F32Range {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if f32::is_nan(self.step) {
            panic!("step is nan");
        }
        if (self.now - self.end) * self.step > 0.0 {
            return None;
        }
        let var = self.now;
        self.now = self.now + self.step;
        return Some(var);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Triangle {
    a: Vec2,
    b: Vec2,
    c: Vec2,
}

fn f32_cmp(a: f32, b: f32) -> Ordering {
    if a > b {
        return Ordering::Greater;
    } else if a < b {
        return Ordering::Less;
    }
    return Ordering::Equal;
}

impl Triangle {
    pub fn new(a: Vec2, b: Vec2, c: Vec2) -> Triangle {
        let mut args = [a, b, c];
        args.sort_by(|v1, v2| f32_cmp(v1.y, v2.y));
        args.reverse();
        Triangle {
            a: args[0],
            b: args[1],
            c: args[2],
        }
    }

    pub fn into_iter(&self) -> TriangleIterator {
        TriangleIterator {
            triangle: *self,
            lines: None,
            state: 0,
            scanner: None,
        }
    }
}

pub fn iter_triangle<F: FnMut(Vec2)>(trig: &Triangle, mut f: F) {
    let a = trig.a;
    let b = trig.b;
    let c = trig.c;
    let y0 = a.y;

    let d1 = c - a;
    let d2 = b - a;
    let d3 = c - b;

    let dx1 = d1.x / d1.y;
    let dx2 = d2.x / d2.y;
    let dx3 = d3.x / d3.y;

    // scan upper triangle.
    assert!(b.y >= c.y);
    assert!(a.y >= b.y);
    if a.y > b.y {
        for yi in F32Range::new(y0.floor(), b.y.ceil(), Some(-1.0)) {
            assert!(yi >= b.y);
            let x1 = a.x + dx1 * (yi - a.y);
            let x2 = a.x + dx2 * (yi - a.y);

            for xi in F32Range::new(x1.floor(), x2.ceil(), None) {
                f(Vec2::new(xi, yi));
            }
        }
    }

    if b.y > c.y {
        // scan lower triangle.
        for yi in F32Range::new(b.y.floor(), c.y.ceil(), Some(-1.0)) {
            let x1 = a.x + dx1 * (yi - a.y);
            let x2 = b.x + dx3 * (yi - b.y);
            for xi in F32Range::new(x1.floor(), x2.ceil(), None) {
                f(Vec2::new(xi, yi));
            }
        }
    }
}

#[derive(Debug)]
pub struct TriangleIterator {
    triangle: Triangle,
    lines: Option<(LineYIterator, LineYIterator)>,
    state: i32,
    scanner: Option<LineIterator>,
}

impl Iterator for TriangleIterator {
    type Item = Vec2;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref mut scanner) = self.scanner {
            let n = scanner.next();
            if let Some(n2) = n {
                return Some(n2.clone());
            }
            self.scanner = None;
        }
        if let Some(ref mut l) = self.lines {
            let a = l.0.next();
            let b = l.1.next();
            println!(">> {:?} {:?}", a, b);
            if let Some(a) = a {
                if let Some(b) = b {
                    self.scanner = Some(Line::new(a, b).into_iter());
                    println!("Next line..");
                    return self.next();
                }
            } else {
            }
        }
        if self.state == 0 {
            self.state = 1;
            let a = self.triangle.a;
            let l1 = Line::new(a, self.triangle.b).into_iter_y();
            let l2 = Line::new(a, self.triangle.c).into_iter_y();

            self.lines = Some((l1, l2));
            println!("Next part..");
            return self.next();
        }

        if self.state == 1 {
            self.state = 2;
            let _a = self.triangle.b;
            if let Some(ref b) = self.lines {
                let l2 = Line::new(self.triangle.b, self.triangle.c).into_iter_y();
                println!("Next part2..");

                self.lines = Some((b.0, l2));
                return self.next();
            }
        }
        return Option::None;
    }
}
