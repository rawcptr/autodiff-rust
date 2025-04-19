use std::{fmt::Display, ops::Index};

#[derive(Debug, Clone)]
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

impl Shape {
    pub fn change(&mut self, shape: impl Into<Shape>) {
        let shape = shape.into();
        assert_eq!(shape.0.iter().product::<usize>(), self.0.iter().product());
        self.0 = shape.0;
    }

    #[inline]
    pub fn ndims(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    #[inline]
    pub fn linear_index<const D: usize>(&self, indices: [usize; D]) -> usize {
        debug_assert!(indices.len() == self.0.len());

        indices
            .iter()
            .zip(&self.0)
            .try_fold(0, |acc, (dim, i)| (i < dim).then_some(acc * dim + i))
            .expect("invalid indices")
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.0.len());
        &self.0[index]
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape({:?})", &self.0)
    }
}
