use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::grid_display::GridDisplay;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;

use std::io;

#[allow(unused)]
pub(crate) fn test() {
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("error: unable to read user input");
    let values: Vec<usize> = input
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();

    let matrix: Matrix<f64> = (values[0], values[1])
        .generate_random()
        .mul(&10f64)
        .into_iter()
        .map(|x| x.into_iter().map(|x| x.trunc()).collect())
        .collect();

    let transposed = matrix.transpose();

    matrix.display();
    transposed.display();
}
