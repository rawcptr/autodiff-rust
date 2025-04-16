use crate::shape::Shape;


pub enum TensorError {
    InconsistentDimensions { 
        expected: Shape,
        received: Shape
    }
}