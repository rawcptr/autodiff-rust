use std::{fmt::Display, ops::Index};

pub struct Shape(Vec<usize>);

impl From<(usize, usize, usize)> for Shape {
    fn from(value: (usize, usize, usize)) -> Self {
        Shape(vec![value.0, value.1, value.2])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(value: (usize, usize)) -> Self {
        Shape(vec![value.0, value.1])
    }
}
impl From<usize> for Shape {
    fn from(value: usize) -> Self {
        Shape(vec![value])
    }
}

pub enum Dims {
    One = 1,
    Two = 2,
    Three = 3,
}

impl From<usize> for Dims {
    fn from(value: usize) -> Self {
        use Dims::*;
        match value {
            1 => One,
            2 => Two,
            3 => Three,
            x => panic!("got {x} dims but only three dims are supported"),
        }
    }
}

impl Shape {
    pub fn change(&mut self, shape: impl Into<Shape>) {
        let shape = shape.into();
        assert_eq!(shape.0.iter().product::<usize>(), self.0.iter().product());
        self.0 = shape.0;
    }

    #[inline]
    pub fn ndims(&self) -> Dims {
        self.0.len().into()
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.ndims() as usize);
        &self.0[index]
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.ndims() {
            Dims::One => write!(f, "Shape({})", self[0]),
            Dims::Two => write!(f, "Shape({}, {})", self[0], self[1]),
            Dims::Three => write!(f, "Shape({}, {}, {})", self[0], self[1], self[2]),
        }
    }
}
