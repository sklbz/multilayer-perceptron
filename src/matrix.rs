use rand::prelude::*;

pub type Vector<T> = Vec<T>;
pub type Matrix<T> = Vector<Vector<T>>;
pub type Tensor<T> = Vector<Matrix<T>>;

trait Random {
    fn random(size: (usize, usize, usize)) -> Self;
}

impl Random for Vector<f64> {
    fn random(size: (usize, usize, usize)) -> Self {
        let mut rng = rand::rng();

        let mut vec = Vec::new();
        for _ in 0..size.0 {
            vec.push(rng.random::<f64>());
        }

        vec
    }
}

impl Random for Matrix<f64> {
    fn random(size: (usize, usize, usize)) -> Self {
        let mut matrix = Vector::new();

        for _ in 0..size.0 {
            let row = Vector::random((size.1, 1, 1))
            matrix.push(row);
        }

        matrix
    }
}

impl Random for Tensor<f64> {
    fn random(size: (usize, usize, usize)) -> Self {
        let mut tensor = Vector::new();

        for _ in 0..size.0 {
            let matrix = Matrix::random((size.1, size.2, 1));
            tensor.push(matrix);
        }

        tensor
    }
}
