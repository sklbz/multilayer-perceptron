use crate::linear_algebra::matrix::*;

pub fn parse_params(input: String) -> (Vector<usize>, Tensor<f64>, Matrix<f64>) {
    let binding = input
        .replace("Architecture", "")
        .replace("Weights", "")
        .replace("Biases", "")
        .replace(",", "")
        .replace(":", "");
    let mut lines = binding.lines();

    let architecture: Vector<usize> = lines
        .next()
        .unwrap()
        .replace("[", "")
        .replace("]", "")
        .split_whitespace()
        .map(|s| s.parse::<usize>().unwrap())
        .collect();

    let weights: Tensor<f64> = lines
        .next()
        .unwrap()
        .replace("]", "")
        .split("[")
        .map(|sup| -> Matrix<f64> {
            sup.split("[")
                .map(|sub| sub.split("[").map(|s| s.parse::<f64>().unwrap()).collect())
                .collect::<Matrix<f64>>()
        })
        .collect();

    let biases: Matrix<f64> = lines
        .next()
        .unwrap()
        .replace("]", "")
        .split("[")
        .map(|sub| sub.split("[").map(|s| s.parse::<f64>().unwrap()).collect())
        .collect();

    (architecture, weights, biases)
}
