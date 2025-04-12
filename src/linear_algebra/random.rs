use super::matrix::*;
use rand::prelude::*;

//------------------------------------------------------------------------------

pub trait Random {
    fn random<T: Size>(size: T) -> Self;
}

impl Random for Vector<f64> {
    fn random(size: usize) -> Self {
        let mut rng = rand::rng();

        let mut vec = Vec::new();
        for _ in 0..size {
            vec.push(rng.random::<f64>());
        }

        vec
    }
}

impl Random for Matrix<f64> {
    fn random(size: (usize, usize)) -> Self {
        let mut matrix = Vector::new();

        for _ in 0..size.0 {
            let row = Vector::random(size.1);
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

//------------------------------------------------------------------------------

trait PseudoRandom {
    fn pseudo_random<T: Size>(size: T) -> Self;
}

impl PseudoRandom for Vector<f64> {
    fn pseudo_random(size: usize) -> Self {
        let mut rng = rand::rng();

        let mut vec = Vec::new();
        for _ in 0..size {
            vec.push(rng.random::<f64>());
        }

        vec
    }
}

impl PseudoRandom for Matrix<f64> {
    fn pseudo_random(size: Vec<usize>) -> Self {
        let mut matrix = Vector::new();

        for i in 0..size.len() {
            let row = Vector::pseudo_random(size[i]);
            matrix.push(row);
        }

        matrix
    }
}

impl PseudoRandom for Tensor<f64> {
    fn pseudo_random(size: Matrix<usize>) -> Self {
        let mut tensor = Vector::new();

        for i in 0..size.len() {
            let matrix = Matrix::pseudo_random(size[i]);
            tensor.push(matrix);
        }

        tensor
    }
}
