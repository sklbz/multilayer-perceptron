use crate::matrix::{Matrix, Tensor, Vector};
use std::ops::Mul;

//------------------------------------------------------------------------------

// Scalar product
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

//------------------------------------------------------------------------------

// Dot product
impl Mul<Vector<f64>> for Vector<f64> {
    type Output = f64;

    fn mul(self, other: Vector<f64>) -> Self::Output {
        self.iter().zip(other.iter()).map(|(x, y)| x * y).sum()
    }
}

//------------------------------------------------------------------------------

// Matrix product
impl Mul<Matrix<f64>> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, other: Matrix<f64>) -> Self::Output {}
}
