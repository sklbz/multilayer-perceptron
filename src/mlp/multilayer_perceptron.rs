use super::backpropagation::*;
use super::utils::*;

use crate::linear_algebra::addition::Add;
use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;

pub(crate) struct MultiLayerPerceptron {
    architecture: Vector<usize>,
    weights: Tensor<f64>,
    biases: Matrix<f64>,
}

pub(crate) trait NeuralNetwork {
    fn new(architecture: Vector<usize>) -> Self;

    fn calc(&self, input: Vector<f64>) -> Vector<f64>;

    fn calc_all(&self, input: Vector<f64>) -> Matrix<f64>;

    fn backpropagation(&mut self, database: Database);

    fn gradient(&self, database: Database) -> NeuralNetGradient;

    fn error_function(&self, database: Database) -> f64;

    fn error_gradient(&self, database: Database) -> Vector<f64>;

    fn inner_gradients(&self, database: Database) -> StepwiseGradients;

    fn display(&self);
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
        }
    }

    fn calc(&self, input: Vector<f64>) -> Vector<f64> {
        let mut result = input;

        self.weights
            .iter()
            .zip(self.biases.iter())
            .for_each(|(matrix, bias)| {
                result = matrix.mul(&result).add(bias);
            });

        result.to_vec()
    }

    fn calc_all(&self, input: Vector<f64>) -> Matrix<f64> {
        let mut current = input;

        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(matrix, bias)| -> Vector<f64> {
                current = matrix.mul(&current).add(bias);
                current.clone()
            })
            .collect::<Matrix<f64>>()
    }

    fn backpropagation(&mut self, database: Database) {
        let error = self.error_function(database.clone());

        if error == 0.0 {
            return;
        }

        let NeuralNetGradient {
            weights, biases, ..
        } = self.gradient(database);

        // DEBUG--------------------------------------------------------------------------------------
        println!("\n Weight \n");
        fn size(matrix: &Matrix<f64>) {
            println!("matrix size: {0}x{1}", matrix.len(), matrix[0].len());
            println!("matrix: {:?}", matrix);
        }
        self.weights.iter().for_each(|matrix| size(matrix));
        println!("\n Gradient \n");
        weights.iter().for_each(|matrix| size(matrix));
        // --------------------------------------------------------------------------------------

        self.weights = self.weights.sub(&weights);
        self.biases = self.biases.sub(&biases);
    }

    fn gradient(&self, database: Database) -> NeuralNetGradient {
        let gradients = self.inner_gradients(database);

        let initial_grad = extract_last_layer(gradients.results);

        backprop(
            gradients.weights,
            gradients.activations,
            initial_grad,
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
        let weights: Tensor<f64> = database
            .iter()
            .map(|(input, _, _)| self.calc_all(input.clone()))
            .collect::<Tensor<f64>>()
            .mean()
            .iter()
            .zip(0..self.weights.len())
            .map(|(vec, l): (&Vector<f64>, usize)| -> Matrix<f64> {
                vec.iter()
                    .map(|partial: &f64| -> Vector<f64> {
                        vec![*partial; self.architecture[l + 1]]
                    })
                    .collect::<Matrix<f64>>()
            })
            .collect::<Tensor<f64>>();

        // ∂(unrectified neuron k of layer l+1)
        // ------------------------------------ = 1
        //         ∂(bias k of layer l)

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
}
