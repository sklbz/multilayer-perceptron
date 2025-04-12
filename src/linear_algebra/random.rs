use super::matrix::*;
use rand::prelude::*;

//------------------------------------------------------------------------------

pub trait Random {
    fn random(size: impl Size) -> Self;
}

impl Random for Vector<f64> {
    fn random(size: impl usize) -> Self {
        let mut rng = rand::rng();

        let mut vec = Vec::new();
        for _ in 0..size {
            vec.push(rng.random::<f64>());
        }

        vec
    }
}

impl Random for Matrix<f64> {
    fn random(size: impl (usize, usize)) -> Self {
        let mut matrix = Vector::new();

        for _ in 0..size.0 {
            let row = Vector::random(size.1);
            matrix.push(row);
        }

        matrix
    }
}

impl Random for Tensor<f64> {
    fn random(size: impl (usize, usize, usize)) -> Self {
        let mut tensor = Vector::new();

        for _ in 0..size.0 {
            let matrix = Matrix::random((size.1, size.2, 1));
            tensor.push(matrix);
        }

        tensor
    }
}

//------------------------------------------------------------------------------
