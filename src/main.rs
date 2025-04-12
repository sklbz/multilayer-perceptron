mod linear_algebra;

use crate::linear_algebra::matrix::*;

fn main() {
    let matrix = Matrix.random((3, 2));
    println!("{:?}", matrix);
    println!("{:?}", matrix.transpose());
}
