use crate::matrix::{Matrix, Tensor, Vector};

struct MultiLayerPerceptron {
    architecture: Vec<usize>,
    weights: Tensor<f64>,
    biases: Matrix<f64>,
}

impl MultiLayerPerceptron {
    fn new(architecture: Vec<usize>) -> Self {
        if architecture.len() < 2 {
            panic!("Architecture must have at least 2 layers");
        }

        let layers_count = architecture.len() - 1;

        let colums = architecture.get(..layers_count).unwrap_or(&[]).to_vec();

        let rows = architecture.get(1..).unwrap_or(&[]).to_vec();

        let weights = Tensor::random_dynamic_size((layers_count, rows, colums));
        let biases = Matrix::random_dynamic_size((layers_count, rows));

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
