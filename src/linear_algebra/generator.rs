use super::matrix::*;
use rand::prelude::*;

//------------------------------------------------------------------------------

pub trait Generator<T> {
    fn generate_random(self) -> T;
}

impl Generator<Vector<f64>> for usize {
    fn generate_random(self) -> Vector<f64> {
        let mut rng = rand::rng();

        let mut vec = Vec::new();
        for _ in 0..self {
            vec.push(rng.random::<f64>());
        }

        vec
    }
}

impl Generator<Matrix<f64>> for (usize, usize) {
    fn generate_random(self) -> Matrix<f64> {
        let mut matrix = Vector::new();

        for _ in 0..self.0 {
            let row = self.1.generate_random();
            matrix.push(row);
        }

        matrix
    }
}

impl Generator<Tensor<f64>> for (usize, usize, usize) {
    fn generate_random(self) -> Tensor<f64> {
        let mut tensor = Vector::new();

        for _ in 0..self.0 {
            let matrix = (self.1, self.2).generate_random();
            tensor.push(matrix);
        }

        tensor
    }
}

//---------------------------------------

impl Generator<Matrix<f64>> for Vector<usize> {
    fn generate_random(self) -> Matrix<f64> {
        let mut matrix = Vector::new();

        for i in 0..self.len() {
            let row = self[i].generate_random();
            matrix.push(row);
        }

        matrix
    }
}

impl Generator<Tensor<f64>> for (Vector<usize>, Vector<usize>) {
    fn generate_random(self) -> Tensor<f64> {
        let mut tensor = Vector::new();

        for i in 0..self.0.len() {
            let matrix = (self.0[i], self.1[i]).generate_random();
            tensor.push(matrix);
        }

        tensor
    }
}

impl Generator<Tensor<f64>> for Matrix<usize> {
    fn generate_random(self) -> Tensor<f64> {
        let mut tensor = Vector::new();

        for i in 0..self.len() {
            let matrix = self[i].clone().generate_random();
            tensor.push(matrix);
        }

        tensor
    }
}

//------------------------------------------------------------------------------
