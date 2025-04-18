use crate::linear_algebra::addition::Add;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;

pub fn into_layer(architecture: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let layers_count = architecture.len() - 1;

    let rows: Vec<usize> = architecture.get(1..).unwrap_or(&[]).to_vec();

    let columns: Vec<usize> = architecture.get(..layers_count).unwrap_or(&[]).to_vec();

    (rows, columns)
}

pub fn square_error(result: &Vec<f64>, target: &Vec<f64>) -> f64 {
    result
        .iter()
        .zip(target.iter())
        .map(|(calc, target)| (calc - target).powf(2.0))
        .sum::<f64>()
}

// ------------------------------------------------------------------------------------

pub(super) trait Mean {
    type Output;
    fn mean(&self) -> Self::Output;
}

impl Mean for Tensor<f64> {
    type Output = Matrix<f64>;

    fn mean(&self) -> Self::Output {
        let mut mean = self[0].clone().mul(&0f64);

        self.iter().for_each(|matrix| mean = mean.add(&matrix));

        let multiplier = 1f64 / self.len() as f64;
        mean.mul(&multiplier)
    }
}

impl Mean for Matrix<f64> {
    type Output = Vector<f64>;

    fn mean(&self) -> Self::Output {
        let mut mean = vec![0f64; self[0].len()];

        self.iter().for_each(|vector| mean = mean.add(&vector));
        let multiplier = 1f64 / self.len() as f64;
        mean.mul(&multiplier)
    }
}

impl Mean for Vector<f64> {
    type Output = f64;

    fn mean(&self) -> Self::Output {
        let multiplier = 1f64 / self.len() as f64;
        self.into_iter().sum::<f64>() * multiplier
    }
}

// ------------------------------------------------------------------------------------

pub(super) type Database = Vec<(Vector<f64>, Vector<f64>, f64)>;
pub(super) struct StepwiseGradients {
    pub activations: Tensor<f64>,
    pub weights: Matrix<f64>,
    pub biases: Matrix<f64>,
    pub results: Vector<f64>,
}
