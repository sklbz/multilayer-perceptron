use super::utils::*;
use crate::linear_algebra::addition::Add;
use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;

pub(crate) struct MultiLayerPerceptron {
    weights: Tensor<f64>,
    biases: Matrix<f64>,
}

impl MultiLayerPerceptron {
    pub fn new(architecture: Vector<usize>) -> Self {
        if architecture.len() < 2 {
            panic!("Architecture must have at least 2 layers");
        }

        let (rows, columns): (Vector<usize>, Vector<usize>) = into_layer(architecture);

        let weights = (rows.clone(), columns).generate_random();

        let biases = rows.generate_random();

        Self { weights, biases }
    }

    pub fn calc(&self, input: Vector<f64>) -> Vector<f64> {
        let mut result = input;

        self.weights
            .iter()
            .zip(self.biases.iter())
            .for_each(|(matrix, bias)| {
                result = matrix.mul(&result).add(bias);
            });

        result.to_vec()
    }

    pub fn train(&mut self, database: Vec<(Vector<f64>, Vector<f64>, f64)>) -> () {
        // Neighbors checking based approach : Discrete Gradient Descent
        // First applying this on the weights, and then on the biases

        // Weights
        for step in [-1f64, 0f64, 1f64] {
            let increment = self
                .weights
                .iter()
                .map(|matrix| {
                    matrix
                        .iter()
                        .map(|vector| vector.iter().map(|_| step).collect())
                        .collect()
                })
                .collect();

            println!("Step: {}", step);
            println!("Error: {}", self.error_function(database));
        }

        // Biases
    }

    pub fn backpropagation(&mut self, database: Vec<(Vector<f64>, Vector<f64>, f64)>) -> () {
        let error = self.error_function(database);

        if error == 0.0 {
            return;
        }
    }

    fn error_function(&self, database: Vec<(Vector<f64>, Vector<f64>, f64)>) -> f64 {
        database
            .iter()
            .map(|(input, target, coefficient)| {
                square_error(&self.calc(input.clone()), target) * coefficient
            })
            .sum()
    }

    pub fn display(&self) {
        println!("Weights:");
        println!("{:?}", self.weights);
        println!("Biases:");
        println!("{:?}", self.biases);
    }
}
