#[test]
pub fn test() {
    use crate::linear_algebra::matrix::Matrix;
    use crate::linear_algebra::matrix::Vector;
    use crate::linear_algebra::product::Mul;

    let matrix: Matrix<f64> = vec![vec![1., 2., 1.], vec![3., 4., 1.]];
    let vector: Vector<f64> = vec![1., 2., 3.];
    let result: Vector<f64> = (&matrix).mul(&vector);

    assert_eq!(result, vec![8., 14.]);
}
