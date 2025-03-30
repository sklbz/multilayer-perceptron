use crate::size::Size;
use rand::prelude::*;

pub type Vector<T> = Vec<T>;
pub type Matrix<T> = Vector<Vector<T>>;
pub type Tensor<T> = Vector<Matrix<T>>;

pub type PseudoMatrix<T> = Vector<Vector<T>>;
pub type PseudoTensor<T> = Vector<Matrix<T>>;

//------------------------------------------------------------------------------

pub trait Length {
    fn size(&self) -> impl Size;
}

impl Length for Vector<f64> {
    fn size(&self) -> impl Size {
        self.len()
    }
}

impl Length for Matrix<f64> {
    fn size(&self) -> impl Size {
        (self.len(), self[0].len())
    }
}

impl Length for Tensor<f64> {
    fn size(&self) -> impl Size {
        (self.len(), self[0].len(), self[0][0].len())
    }
}

//------------------------------------------------------------------------------

pub trait Transpose {
    fn transpose(&self) -> Self;
}

impl Transpose for Matrix<f64> {
    fn transpose(&self) -> Self {
        let mut matrix = Vector::new();

        for i in 0..self[0].len() {
            let mut row = Vector::new();
            for j in 0..self.len() {
                row.push(self[j][i]);
            }
            matrix.push(row);
        }

        matrix
    }
}

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

impl Random for PseudoMatrix<f64> {
    fn random(size: (usize, Vec<usize>)) -> Self {
        let mut matrix = Vector::new();

        for i in 0..size.0 {
            let row = Vector::random(size.1[i]);
            matrix.push(row);
        }

        matrix
    }
}

impl Random for PseudoTensor<f64> {
    fn random(size: (usize, Vec<usize>, Vec<usize>)) -> Self {
        let mut tensor = Vector::new();

        for i in 0..size.0 {
            let matrix = Matrix::random((size.1[i], size.2[i]));
            tensor.push(matrix);
        }

        tensor
    }
}
