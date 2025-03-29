use matrix::{Matrix, Tensor, Vector};
use std::ops::Mul;

impl Mul<f64> for Vector<f64> {
    type Output = Vector<f64>;

    fn mul(self, scalar: f64) -> Self::Output {
        self.iter().map(|x| x * scalar).collect()
    }
}

impl Mul<f64> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, scalar: f64) -> Self::Output {
        self.iter().map(|x| x * scalar).collect()
    }
}

impl Mul<f64> for Tensor<f64> {
    type Output = Tensor<f64>;

    fn mul(self, scalar: f64) -> Self::Output {
        self.iter().map(|x| x * scalar).collect()
    }
}
