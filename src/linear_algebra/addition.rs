use crate::linear_algebra::matrix::*;

pub(crate) trait Add<T> {
    type Output;
    fn add(&self, other: &T) -> Self::Output;
    fn sub(&self, other: &T) -> Self::Output;
}

impl Add<Vector<f64>> for Vector<f64> {
    type Output = Vector<f64>;

    fn add(&self, other: &Vector<f64>) -> Self::Output {
        self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
    }

    fn sub(&self, other: &Vector<f64>) -> Self::Output {
        self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }
}

impl Add<Matrix<f64>> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn add(&self, other: &Matrix<f64>) -> Self::Output {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.add(b))
            .collect()
    }

    fn sub(&self, other: &Matrix<f64>) -> Self::Output {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.sub(b))
            .collect()
    }
}

impl Add<Tensor<f64>> for Tensor<f64> {
    type Output = Tensor<f64>;

    fn add(&self, other: &Tensor<f64>) -> Self::Output {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.add(b))
            .collect()
    }

    fn sub(&self, other: &Tensor<f64>) -> Self::Output {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.sub(b))
            .collect()
    }
}
