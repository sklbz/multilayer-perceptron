use crate::linear_algebra::{grid_display::GridDisplay, matrix::*, product::Mul};

#[allow(unused)]
pub fn test() {
    let matrix: Matrix<f64> = vec![vec![1., 2., 1.], vec![3., 4., 1.]];
    let vector: Vector<f64> = vec![1., 2., 3.];
    let result: Vector<f64> = (&matrix).mul(&vector);

    matrix.display();
    vec![vector].display();
    vec![result].display();
}
