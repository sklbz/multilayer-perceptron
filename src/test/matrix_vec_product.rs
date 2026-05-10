use crate::linear_algebra::{
    matrix::{Matrix, Vector},
    product::Mul,
};

#[test]
pub fn test() {
    let matrix: Matrix<f64> = vec![vec![1., 2., 1.], vec![3., 4., 1.]];
    let vector: Vector<f64> = vec![1., 2., 3.];
    let result: Vector<f64> = (&matrix).mul(&vector);

    assert_eq!(result, vec![8., 14.]);
}
