use crate::matrix::{Matrix, Tensor, Vector};

struct MultiLayerPerceptron {
    architecture: Vec<u32>,
    weights: Tensor<f64>,
    biases: Matrix<f64>,
}

impl MultiLayerPerceptron {
    fn new(architecture: Vec<u32>) -> Self {
        if architecture.len() < 2 {
            panic!("Architecture must have at least 2 layers");
        }

        let weights = Tensor::new();
        let biases = Matrix::new();

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
            .into_iter()
            .for_each(|(matrix, bias)| {
                result = matrix * result + bias;
            });

        result
    }

    fn train(&mut self, database: Vec<(f64, f64)>) -> () {}
}
