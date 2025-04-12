pub(crate) use crate::linear_algebra::size::Size;

pub type Vector<T> = Vec<T>;
pub type Matrix<T> = Vector<Vector<T>>;
pub type Tensor<T> = Vector<Matrix<T>>;

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
