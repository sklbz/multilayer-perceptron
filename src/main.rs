mod matrix;
mod matrix_product;
mod multilayer_perceptron;
mod size;

use matrix::Random;
use matrix::Transpose;

fn main() {
    let matrix = matrix::Matrix::random((3, 2));
    println!("{:?}", matrix);
    println!("{:?}", matrix.transpose());
}
