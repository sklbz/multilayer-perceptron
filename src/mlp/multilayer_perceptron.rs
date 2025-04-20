use super::backpropagation::*;
use super::partial_gradient::weight_partial;
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
                let value = current.clone();
                current = matrix.mul(&current).add(bias);
                value
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

        /* // DEBUG------------------------------------------------------------------------------
                println!(
                    "---------------------------------------------------------------------------------------------------------------------------------------------------"
                );
                println!("DEBUG");
                println!();
                for (weight, grad) in self.weights.iter().zip(weights.iter()) {
                    println!("Weight size: {}x{}", weight.len(), weight[0].len());
                    println!("Weight gradient size: {}x{}", grad.len(), grad[0].len());
                }
                println!();
                println!(
                    "---------------------------------------------------------------------------------------------------------------------------------------------------"
                );
                // ------------------------------------------------------------------------------
        */
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
