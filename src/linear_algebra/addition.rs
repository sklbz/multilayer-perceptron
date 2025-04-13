use crate::linear_algebra::matrix::*;

pub(crate) trait Add<T> {
    type Output;
    fn add(&self, other: &T) -> Self::Output;
}

impl Add<Vector<f64>> for Vector<f64> {
    type Output = Vector<f64>;
    fn add(&self, other: &Vector<f64>) -> Self::Output {
        self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
    }
}
