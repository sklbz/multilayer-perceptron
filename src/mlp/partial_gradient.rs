use crate::linear_algebra::matrix::*;

use super::utils::*;

pub fn weight_partial(
    calc: &dyn Fn(Vector<f64>) -> Matrix<f64>,
    architecture: &Vector<usize>,
    database: Database,
) -> Tensor<f64> {
    let layer_count = architecture.len();

    database
        .iter()
        .map(|(input, _, _)| calc(input.clone()))
        .collect::<Tensor<f64>>()
        .mean()
        .iter()
        .zip(0..layer_count)
        .map(|(vec, l): (&Vector<f64>, usize)| -> Matrix<f64> {
            vec.iter()
                .map(|partial: &f64| -> Vector<f64> { vec![*partial; architecture[l + 1]] })
                .collect::<Matrix<f64>>()
        })
        .collect::<Tensor<f64>>()
}
