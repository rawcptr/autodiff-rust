use crate::shape::Shape;

#[derive(Debug, Clone)]
pub enum TensorError {
    InconsistentDims { expected: Shape, received: Shape },
    Memory(String),
    Broadcast { d1: usize, d2: usize },
    InvalidOp(String),
}

impl TensorError {
    pub fn inconsistent(expected: &[usize], received: &[usize]) -> Self {
        Self::InconsistentDims {
            expected: Shape::from(expected),
            received: Shape::from(received),
        }
    }
}

impl std::error::Error for TensorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::InconsistentDims { expected, received } => {
                write!(
                    f,
                    "inconsistent dimensions. expected: {expected}, received: {received}"
                )
            }
            TensorError::Memory(why) => {
                write!(f, "memory handling violation: {why}")
            }
            TensorError::InvalidOp(err) => {
                write!(f, "invalid operation: {err}")
            }
            TensorError::Broadcast { d1: dim1, d2: dim2 } => {
                write!(f, "cannot broadcast dimensions: {dim1} vs {dim2}")
            }
        }
    }
}
