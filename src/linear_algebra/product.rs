use crate::linear_algebra::matrix::*;

pub(crate) trait Mul<T> {
    type Output;

    fn mul(self, other: &T) -> Self::Output;
}

//------------------------------------------------------------------------------

// Scalar product
impl Mul<f64> for Vector<f64> {
    type Output = Vector<f64>;

    fn mul(self, scalar: &f64) -> Self::Output {
        self.iter().map(|x| x * scalar).collect()
    }
}

impl Mul<f64> for &Vector<f64> {
    type Output = Vector<f64>;

    fn mul(self, scalar: &f64) -> Self::Output {
        self.iter().map(|x| x * scalar).collect()
    }
}

impl Mul<f64> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, scalar: &f64) -> Self::Output {
        self.iter().map(|x: &Vector<f64>| x.mul(scalar)).collect()
    }
}

impl Mul<f64> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, scalar: &f64) -> Self::Output {
        self.iter().map(|x: &Vector<f64>| x.mul(scalar)).collect()
    }
}

impl Mul<f64> for Tensor<f64> {
    type Output = Tensor<f64>;

    fn mul(self, scalar: &f64) -> Self::Output {
        self.iter().map(|x: &Matrix<f64>| x.mul(scalar)).collect()
    }
}

//------------------------------------------------------------------------------
// Dot product

impl Mul<Vector<f64>> for Vector<f64> {
    type Output = f64;

    fn mul(self, other: &Vector<f64>) -> Self::Output {
        if self.len() != other.len() {
            println!();
            println!("self.len() != other.len()");
            println!("{} != {}", self.len(), other.len());
            panic!("Attempt to multiply two vectors with different length");
        }

        self.iter().zip(other.iter()).map(|(x, y)| x * y).sum()
    }
}

impl Mul<Vector<f64>> for &Vector<f64> {
    type Output = f64;
    fn mul(self, other: &Vector<f64>) -> Self::Output {
        if self.len() != other.len() {
            println!();
            println!("self.len() != other.len()");
            println!("{} != {}", self.len(), other.len());
            panic!("Attempt to multiply two vectors with different length");
        }

        self.iter().zip(other.iter()).map(|(x, y)| x * y).sum()
    }
}

//------------------------------------------------------------------------------
// Matrix product

impl Mul<Vector<f64>> for Matrix<f64> {
    type Output = Vector<f64>;

    fn mul(self, other: &Vector<f64>) -> Self::Output {
        self.iter().map(|x| x.mul(other)).collect()
    }
}

impl Mul<Vector<f64>> for &Matrix<f64> {
    type Output = Vector<f64>;
    fn mul(self, other: &Vector<f64>) -> Self::Output {
        self.iter().map(|x: &Vector<f64>| x.mul(other)).collect()
    }
}

impl Mul<Matrix<f64>> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, _other: &Matrix<f64>) -> Self::Output {
        todo!("Matrix multiplication");
    }
}

//------------------------------------------------------------------------------
impl Mul<Vector<f64>> for Tensor<f64> {
    type Output = Matrix<f64>;

    fn mul(self, _other: &Vector<f64>) -> Self::Output {
        self[0].clone()
    }
}
