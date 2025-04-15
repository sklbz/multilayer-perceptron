use crate::linear_algebra::addition::Add;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;

pub fn into_layer(architecture: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let layers_count = architecture.len() - 1;

    let rows: Vec<usize> = architecture
        .get(1..)
        .unwrap_or(&[])
        .to_vec()
        .into_iter()
        .map(|layer_size| layer_size as usize)
        .collect();

    let columns: Vec<usize> = architecture
        .get(..layers_count)
        .unwrap_or(&[])
        .to_vec()
        .into_iter()
        .map(|layer_size| layer_size as usize)
        .collect();

    (rows, columns)
}

pub fn square_error(result: &Vec<f64>, target: &Vec<f64>) -> f64 {
    result
        .iter()
        .zip(target.iter())
        .map(|(calc, target)| (calc - target).powf(2.0))
        .sum::<f64>()
}

pub fn mean(this: &Tensor<f64>) -> Matrix<f64> {
    let mut mean = this[0].clone().add(&this[0].clone().mul(&-1f64));

    this.iter().for_each(|matrix| mean = mean.add(&matrix));

    let multiplier = 1f64 / this.len() as f64;
    mean.mul(&multiplier)
}
