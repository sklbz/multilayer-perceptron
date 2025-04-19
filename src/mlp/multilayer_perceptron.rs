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

        type Gradient<T> = T;
        struct NeuralNetGradient {
            pub neurons: Gradient<Matrix<f64>>,
            pub weights: Gradient<Tensor<f64>>,
            pub biases: Gradient<Matrix<f64>>,
        }

        let gradients = self.inner_gradients(database);

        //TODO: test all this mess
        fn backprop(
            chain: StepwiseGradients,
            grad: NeuralNetGradient,
            depth: usize,
        ) -> NeuralNetGradient {
            if depth == 0 {
                return grad;
            }

            let weight: &Matrix<f64> = &chain.weights[depth];
            let activation: &Matrix<f64> = &chain.activations[depth];

            //                       ∂ cost
            // previous_layer[k] = -----------
            //                     ∂(neuron k)
            let previous_layer: Vector<f64> = activation.mul(&grad.neurons[0].clone());

            //                           ∂ cost
            // previous_layer[k, j] = ------------
            //                        ∂(weight kj)
            let previous_weights = weight
                .iter()
                .zip(grad.neurons[0].iter())
                .map(|(w, n): (&Vector<f64>, &f64)| w.mul(n))
                .collect();

            //                       ∂ cost
            // previous_biases[k] = ---------
            //                      ∂(bias k)
            let previous_biases = grad.neurons[0].clone();

            // THIS IS WEIRD
            // I shouldn't have to wrap with vec![]
            let neurons = grad.neurons.prepend(previous_layer);

            let weights = grad.weights.prepend(previous_weights);

            let biases = grad.biases.prepend(previous_biases);

            let extended_grad = NeuralNetGradient {
                neurons,
                weights,
                biases,
            };

            backprop(chain, extended_grad, depth - 1)
        }

        let initial_grad = NeuralNetGradient {
            neurons: vec![gradients.results.clone()],
            weights: vec![],
            biases: vec![],
        };

        let _grad = backprop(gradients, initial_grad, self.weights.len());
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

        //                ∂(unrectified neuron k of layer l+1)
        // biases[l, k] = ------------------------------------ = 1
        //                        ∂(bias k of layer l)

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
