use crate::{error::TensorError, shape::Shape, storage::Storage};

pub trait Tensorizable<T> {
    fn to_tensor(self) -> Result<Tensor<T>, TensorError>;
}

impl<T> Tensorizable<T> for Vec<T> {
    fn to_tensor(self) -> Result<Tensor<T>, TensorError> {
        let storage = Storage::<T>::new(self.len());
        let shape = Shape::from(1);

        Ok(Tensor {
            storage,
            shape,
            requires_grad: false,
            grad: None,
        })
    }
}

impl<T> Tensorizable<T> for Vec<Vec<T>> {
    fn to_tensor(self) -> Result<Tensor<T>, TensorError> {
        let (dim0, dim1) = (self.len(), self[0].len());
        if let Some(row) = self.iter().find(|x| x.len() != dim1) {
            let expected = (self.len(), dim1).into();
            let received = (self.len(), row.len()).into();
            return Err(TensorError::InconsistentDimensions { expected, received });
        }

        let buf: Vec<T> = self.into_iter().flatten().collect();
        let mut storage = Storage::<T>::new(buf.len());
        storage.initialize_from_iter(buf);

        Ok(Tensor {
            storage,
            shape: (dim0, dim1).into(),
            requires_grad: false,
            grad: None,
        })
    }
}

fn check_dims_3d<T>(data: &[Vec<Vec<T>>]) -> Result<Shape, TensorError> {
    if data.is_empty() {
        return Ok((0, 0, 0).into());
    }
    let planes = data.len();

    let expected_rows = data[0].len();
    let expected_columns = data[0].first().map_or(0, Vec::len);

    for plane in data.iter() {
        let actual_rows = plane.len();
        if actual_rows != expected_rows {
            return Err(TensorError::InconsistentDimensions {
                expected: (planes, expected_rows, expected_columns).into(),
                received: (planes, actual_rows, expected_columns).into(),
            });
        }

        for row in plane.iter() {
            let actual_columns = row.len();
            if actual_columns != expected_columns {
                return Err(TensorError::InconsistentDimensions {
                    expected: (planes, expected_rows, expected_columns).into(),
                    received: (planes, actual_rows, actual_columns).into(),
                });
            }
        }
    }

    Ok((planes, expected_rows, expected_columns).into())
}

impl<T> Tensorizable<T> for Vec<Vec<Vec<T>>> {
    fn to_tensor(self) -> Result<Tensor<T>, TensorError> {
        let shape = check_dims_3d(&self)?;

        // initialize storage
        let buf: Vec<_> = self
            .into_iter()
            .flat_map(|v| v.into_iter().flatten())
            .collect();
        
        let mut storage = Storage::<T>::new(buf.len());

        storage.initialize_from_iter(buf);

        Ok(Tensor {
            storage,
            shape,
            requires_grad: false,
            grad: None,
        })
    }
}

pub struct Tensor<T> {
    // we want to _loan_ this tensor out
    storage: Storage<T>,
    shape: Shape,
    requires_grad: bool,
    grad: Option<Storage<T>>,
}

impl<T> Tensor<T> {
    
    // copies the elements from the container into the
    // pub fn new(container: Vec<T>) -> Self {
    //     let mut storage = Storage::<T>::new(container.len());

    //     let shape: Shape = shape.into();
    //     storage.initialize_from_iter(container);

    //     Self {
    //         storage,
    //         shape,
    //         requires_grad: false,
    //         grad: None,
    //     }
    // }
}
