use super::activation::rectified_linear::RELU;
use super::backpropagation::*;
use super::io::parse_params;
use super::partial_gradient::weight_partial;
use super::utils::*;

use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;
use crate::{linear_algebra::addition::Add, mlp::activation_function::Activation};

use std::fs::File;
use std::io;
use std::path::Path;

pub struct MultiLayerPerceptron {
    architecture: Vector<usize>,
    weights: Tensor<f64>,
    biases: Matrix<f64>,
    activation: &'static Activation,
}

impl MultiLayerPerceptron {
    pub fn zeroed(architecture: Vector<usize>) -> Self {
        let (rows, columns): (Vector<usize>, Vector<usize>) = into_layer(&architecture);

        // Matrices de poids remplies de 0
        let weights: Tensor<f64> = rows
            .iter()
            .zip(columns.iter())
            .map(|(r, c)| vec![vec![0.0; *c]; *r])
            .collect();

        // Bias remplis de 0
        let biases: Matrix<f64> = rows.iter().map(|r| vec![0.0; *r]).collect();

        Self {
            architecture,
            weights,
            biases,
            activation: &RELU,
        }
    }
}

pub trait NeuralNetwork {
    fn new(architecture: Vector<usize>) -> Self;

    fn calc(&self, input: Vector<f64>) -> Vector<f64>;

    fn calc_all(&self, input: Vector<f64>) -> Matrix<f64>;

    fn backpropagation(&mut self, database: Database, iterations: usize, momentum: f64);

    fn gradient(&self, database: Database) -> NeuralNetGradient;

    fn error_function(&self, database: Database) -> f64;

    fn error_gradient(&self, database: Database) -> Vector<f64>;

    fn inner_gradients(&self, database: Database) -> StepwiseGradients;

    fn display(&self);

    fn params(&self) -> String;

    fn load_params(&mut self, file_path: &str);
}

impl NeuralNetwork for MultiLayerPerceptron {
    fn new(architecture: Vector<usize>) -> Self {
        if architecture.len() < 2 {
            panic!("Architecture must have at least 2 layers");
        }

        let (rows, columns): (Vector<usize>, Vector<usize>) = into_layer(&architecture);

        let weights = (rows.clone(), columns).generate_random();

        let biases = rows.generate_random();

        Self {
            architecture,
            weights,
            biases,
            activation: &RELU,
        }
    }

    fn calc(&self, input: Vector<f64>) -> Vector<f64> {
        let mut result = input;

        self.weights
            .iter()
            .zip(self.biases.iter())
            .for_each(|(matrix, bias)| {
                result = self.activation.apply(matrix.mul(&result).add(bias));
            });

        result.to_vec()
    }

    fn calc_all(&self, input: Vector<f64>) -> Matrix<f64> {
        let mut current = input;

        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(matrix, bias)| -> Vector<f64> {
                let pre_activation = matrix.mul(&current).add(bias);
                let value = current.clone();
                current = self.activation.apply(pre_activation);
                value
            })
            .collect::<Matrix<f64>>()
    }

    fn backpropagation(&mut self, database: Database, iterations: usize, momentum: f64) {
        for _ in 0..iterations {
            let error = self.error_function(database.clone());

            if error == 0.0 {
                return;
            }

            let NeuralNetGradient {
                weights, biases, ..
            } = self.gradient(database.clone());

            self.weights = self.weights.sub(&weights.mul(&momentum));
            self.biases = self.biases.sub(&biases.mul(&momentum));
        }
    }

    fn gradient(&self, database: Database) -> NeuralNetGradient {
        let gradients = self.inner_gradients(database);

        let initial_grad = extract_last_layer(gradients.results);

        backprop(
            gradients.weights,
            gradients.activations,
            initial_grad,
            &self.architecture,
            self.weights.len(),
        )
    }

    fn inner_gradients(&self, database: Database) -> StepwiseGradients {
        //                        ∂(unrectified neuron k of layer l+1)
        // activations[l, k, j] = ------------------------------------
        //                               ∂(neuron j of layer l)
        let activations: Tensor<f64> = self.weights.clone();

        //                 ∂(unrectified neuron k of layer l+1)
        // weights[l, k, j] = ------------------------------------
        //                       ∂(weight kj of layer l)
        let weights: Tensor<f64> = weight_partial(
            &|input: Vector<f64>| -> Matrix<f64> { self.calc_all(input) },
            &self.architecture,
            database.clone(),
        );

        // ∂(unrectified neuron k of layer l+1)
        // ------------------------------------ = 1
        //         ∂(bias k of layer l)

        //       ∂(neuron k of layer l+1)
        // ------------------------------------ = 1
        // ∂(unrectified neuron k of layer l+1)
        let rectification = 0;

        //                      ∂ cost
        // results[k] = -------------------------
        //              ∂(neuron k of last layer)
        let results: Vector<f64> = self.error_gradient(database);

        StepwiseGradients {
            activations,
            weights,
            results,
        }
    }

    fn error_function(&self, database: Database) -> f64 {
        database
            .iter()
            .map(|(input, target, coefficient)| {
                square_error(&self.calc(input.clone()), target) * coefficient
            })
            .sum()
    }

    fn error_gradient(&self, database: Database) -> Vector<f64> {
        database
            .iter()
            .map(|(input, target, coefficient)| -> Vector<f64> {
                self.calc(input.clone())
                    .add(&target.mul(&-1f64))
                    .mul(&(2f64 * coefficient))
            })
            .collect::<Matrix<f64>>()
            .mean()
    }

    fn display(&self) {
        println!("Weights:");
        println!("{:?}", self.weights);
        println!("Biases:");
        println!("{:?}", self.biases);
    }

    fn params(&self) -> String {
        format!(
            "Architecture: {:?}\nWeights: {:?}\nBiases: {:?}",
            self.architecture, self.weights, self.biases
        )
    }

    fn load_params(&mut self, file_path: &str) {
        let path = Path::new(file_path);
        let file = File::open(path).unwrap();
        let content = io::read_to_string(file).unwrap();
        let (architecture, weights, biases) = parse_params(content);
        self.architecture = architecture;
        self.weights = weights;
        self.biases = biases;
    }
}
