use crate::shape::Shape;

#[derive(Debug, Clone)]
pub enum TensorError {
    InconsistentDimensions { expected: Shape, received: Shape },
    MemoryViolation { why: String },
}

impl std::error::Error for TensorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TensorError::InconsistentDimensions { .. } | TensorError::MemoryViolation { .. } => {
                None
            }
        }
    }
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::InconsistentDimensions { expected, received } => {
                write!(
                    f,
                    "inconsistent dimensions. expected: {expected}, received: {received}"
                )
            }
            TensorError::MemoryViolation { why } => {
                write!(f, "memory handling violation: {why}")
            }
        }
    }
}
