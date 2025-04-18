use crate::linear_algebra::matrix::*;
use crate::linear_algebra::size::Size;

#[allow(unused)]
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
