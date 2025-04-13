use crate::linear_algebra::addition::Add;
use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::grid_display::GridDisplay;
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

        let layers_count = architecture.len() - 1;

        let colums: Vector<usize> = architecture
            .get(..layers_count)
            .unwrap_or(&[])
            .to_vec()
            .into_iter()
            .map(|layer_size| layer_size as usize)
            .collect();

        let rows: Vector<usize> = architecture
            .get(1..)
            .unwrap_or(&[])
            .to_vec()
            .into_iter()
            .map(|layer_size| layer_size as usize)
            .collect();

        let weights = (rows.clone(), colums.clone()).generate_random();

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

        fn calculate_error(
            weights: Tensor<f64>,
            biases: Matrix<f64>,
            input: Vector<f64>,
            target: Vector<f64>,
            coefficient: f64,
        ) -> f64 {
            let mut calc = input;
            weights
                .iter()
                .zip(biases.iter())
                .for_each(|(matrix, bias)| {
                    calc = matrix.mul(&calc).add(bias);
                });

            let error = calc
                .iter()
                .zip(target.iter())
                .fold(0f64, |acc, (calc, target)| acc + (calc - target).powf(2.0));

            error * coefficient
        }

        let iterations = 10;

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
            println!(
                "Error: {}",
                calculate_error(
                    self.weights.add(&increment),
                    self.biases.clone(),
                    database[0].0.clone(),
                    database[0].1.clone(),
                    database[0].2
                )
            );
        }

        // Biases
    }

    pub fn display(&self) {
        println!("Weights:");
        println!("{:?}", self.weights);
        println!("Biases:");
        println!("{:?}", self.biases);
    }
}
