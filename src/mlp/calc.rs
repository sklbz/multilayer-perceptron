use crate::linear_algebra::addition::*;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::*;

pub(super) fn update_value(
    current: &mut Vector<f64>,
    weight: &Matrix<f64>,
    bias: &Vector<f64>,
) -> Vector<f64> {
    let result = current.clone();

    *current = weight.mul(current).add(bias);

    result
}
