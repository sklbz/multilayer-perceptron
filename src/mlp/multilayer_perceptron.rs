use super::utils::*;
use crate::linear_algebra::addition::Add;
use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;

pub(crate) struct MultiLayerPerceptron {
    layer_count: usize,
    weights: Tensor<f64>,
    biases: Matrix<f64>,
}

pub(crate) trait NeuralNetwork {
    fn new(architecture: Vector<usize>) -> Self;

    fn calc(&self, input: Vector<f64>) -> Vector<f64>;

    fn calc_all(&self, input: Vector<f64>) -> Matrix<f64>;

    fn backpropagation(&mut self, database: Database);

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

        let layer_count = architecture.len();

        let (rows, columns): (Vector<usize>, Vector<usize>) = into_layer(architecture.clone());

        let weights = (rows.clone(), columns).generate_random();

        let biases = rows.generate_random();

        Self {
            layer_count,
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

        type Gradient<T> = T;
        struct NeuralNetGradient {
            pub neurons: Gradient<Matrix<f64>>,
            pub weights: Gradient<Tensor<f64>>,
            pub biases: Gradient<Matrix<f64>>,
        }

        let gradients = self.inner_gradients(database);

        // Chain rule
        // I could inspire myself from adjacency matrices in graphs to create an elegant algorithm
        // utilizing only a few matrix/tensor multiplication
        // One problem is that I am unsure about the behaviour of the tensor product I defined
        // Not to mention they aren't even proper matrices nor proper tensors
        // I should also expand the weight grad instead of collapsing redondant values

        fn backprop(grad: NeuralNetGradient, depth: usize) -> NeuralNetGradient {
            if depth == 0 {
                return grad;
            }

            //                       ∂ cost
            // previous_layer[k] = -----------
            //                     ∂(neuron k)
            let previous_layer = grad.neurons[0].clone();

            //                           ∂ cost
            // previous_layer[k, j] = ------------
            //                        ∂(weight kj)
            let previous_weights = grad.weights[0].clone();

            //                       ∂ cost
            // previous_biases[k] = ---------
            //                      ∂(bias k)
            let previous_biases = grad.biases[0].clone();

            let neurons = [&[previous_layer], &grad.neurons[..]].concat();

            let weights = [&[previous_weights], &grad.weights[..]].concat();

            let biases = [&[previous_biases], &grad.biases[..]].concat();

            let extended_grad = NeuralNetGradient {
                neurons,
                weights,
                biases,
            };

            backprop(extended_grad, depth - 1)
        }

        let initial_grad = NeuralNetGradient {
            neurons: vec![gradients.results],
            weights: vec![],
            biases: vec![],
        };

        let grad = backprop(initial_grad, self.layer_count);
    }

    fn inner_gradients(&self, database: Database) -> StepwiseGradients {
        //                        ∂(unrectified neuron k of layer l+1)
        // activations[l, k, j] = ------------------------------------
        //                               ∂(neuron j of layer l)
        let activations: Tensor<f64> = self.weights.clone();

        //                 ∂(unrectified neuron k of layer l+1)
        // weights[l, j] = ------------------------------------
        //                       ∂(weight kj of layer l)
        let weights: Matrix<f64> = database
            .iter()
            .map(|(input, _, _)| self.calc_all(input.clone()))
            .collect::<Tensor<f64>>()
            .mean();

        //                ∂(unrectified neuron k of layer l+1)
        // biases[l, k] = ------------------------------------
        //                          ∂(bias k of layer l)
        let biases: Matrix<f64> = self
            .biases
            .iter()
            .map(|vec| vec.iter().map(|_| 1f64).collect())
            .collect();

        //                      ∂ cost
        // results[k] = -------------------------
        //              ∂(neuron k of last layer)
        let results: Vector<f64> = self.error_gradient(database);

        StepwiseGradients {
            activations,
            weights,
            biases,
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
