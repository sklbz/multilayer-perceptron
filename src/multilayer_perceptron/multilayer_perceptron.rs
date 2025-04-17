use super::utils::*;
use crate::linear_algebra::addition::Add;
use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;

pub(crate) struct MultiLayerPerceptron {
    weights: Tensor<f64>,
    biases: Matrix<f64>,
}

pub(crate) trait NeuralNetwork {
    fn new(architecture: Vector<usize>) -> Self;

    fn calc(&self, input: Vector<f64>) -> Vector<f64>;

    fn calc_all(&self, input: Vector<f64>) -> Matrix<f64>;

    fn backpropagation(&mut self, database: Vec<(Vector<f64>, Vector<f64>, f64)>);

    fn error_function(&self, database: Vec<(Vector<f64>, Vector<f64>, f64)>) -> f64;

    fn error_gradient(&self, database: Vec<(Vector<f64>, Vector<f64>, f64)>) -> Vector<f64>;

    fn display(&self);
}

impl NeuralNetwork for MultiLayerPerceptron {
    fn new(architecture: Vector<usize>) -> Self {
        if architecture.len() < 2 {
            panic!("Architecture must have at least 2 layers");
        }

        let (rows, columns): (Vector<usize>, Vector<usize>) = into_layer(architecture);

        let weights = (rows.clone(), columns).generate_random();

        let biases = rows.generate_random();

        Self { weights, biases }
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

    fn backpropagation(&mut self, database: Vec<(Vector<f64>, Vector<f64>, f64)>) {
        let error = self.error_function(database.clone());

        if error == 0.0 {
            return;
        }

        type Gradient<T> = T;

        //                        ∂(non-linearised neuron k of layer l+1)
        // activations[l, k, j] = ---------------------------------------
        //                                ∂(neuron j of layer l)
        let _activations: &Gradient<Tensor<f64>> = &self.weights;

        //                    ∂(non-linearised neuron k of layer l+1)
        // weights[l, k, j] = ---------------------------------------
        //                            ∂(weight kj of layer l)
        let _weights: Gradient<Matrix<f64>> = database
            .iter()
            .map(|(input, _, _)| self.calc_all(input.clone()))
            .collect::<Tensor<f64>>()
            .mean();

        //                      ∂ cost
        // results[k] = -------------------------
        //              ∂(neuron k of last layer)
        let _results: Gradient<Vector<f64>> = self.error_gradient(database);
    }

    fn error_function(&self, database: Vec<(Vector<f64>, Vector<f64>, f64)>) -> f64 {
        database
            .iter()
            .map(|(input, target, coefficient)| {
                square_error(&self.calc(input.clone()), target) * coefficient
            })
            .sum()
    }

    fn error_gradient(&self, database: Vec<(Vector<f64>, Vector<f64>, f64)>) -> Vector<f64> {
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
