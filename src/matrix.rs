use rand::prelude::*;
use std::ops::Mul;

pub type Vector<T> = Vec<T>;
pub type Matrix<T> = Vector<Vector<T>>;
pub type Tensor<T> = Vector<Matrix<T>>;

pub type PseudoMatrix<T> = Vector<Vector<T>>;
pub type PseudoTensor<T> = Vector<Matrix<T>>;

trait Random {
    fn random_static_size(size: (usize, usize, usize)) -> Self;
    fn random_dynamic_size(size: (usize, Vec<usize>, Vec<usize>)) -> Self;
   
}

impl Random for Vector<f64> {
    fn random_static_size(size: (usize, usize, usize)) -> Self {
        let mut rng = rand::rng();

        let mut vec = Vec::new();
        for _ in 0..size.0 {
            vec.push(rng.random::<f64>());
        }

        vec
    }

    // This is actually the exact same but I don't know how to collapse them
    fn random_dynamic_size(size: (usize, Vec<usize>, Vec<usize>)) -> Self {
        let mut rng = rand::rng();

        let mut vec = Vec::new();
        for _ in 0..size.0 {
            vec.push(rng.random::<f64>());
        }

        vec
    }
}

impl Random for Matrix<f64> {
    fn random_static_size(size: (usize, usize, usize)) -> Self {
        let mut matrix = Vector::new();

        for _ in 0..size.0 {
            let row = Vector::random_static_size((size.1, _, _))
            matrix.push(row);
        }

        matrix
    }

    fn random_dynamic_size(size: (usize, Vec<usize>, Vec<usize>)) -> Self {
        let mut matrix = Vector::new();

        for i in 0..size.0 {
            let row = Vector::random_static_size((size.1[i], 1, 1));
            matrix.push(row);
        }

        matrix
    }
}

impl Random for Tensor<f64> {
    fn random_static_size(size: (usize, usize, usize)) -> Self {
        let mut tensor = Vector::new();

        for _ in 0..size.0 {
            let matrix = Matrix::random_static_size((size.1, size.2, 1));
            tensor.push(matrix);
        }

        tensor
    }

    fn random_dynamic_size(size: (usize, Vec<usize>, Vec<usize>)) -> Self {
        let mut tensor = Vector::new();

        for i in 0..size.0 {
            let matrix = Matrix::random_static_size((size.1[i], size.2[i], 1));
            tensor.push(matrix);
        }
}
