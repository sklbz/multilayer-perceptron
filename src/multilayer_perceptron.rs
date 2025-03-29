use matrix::Matrix;

struct MultiLayerPerceptron {
    architecture: Vec<u32>,
    // Should I reduce all the operations to a single matrix multiplication or not ?
    // Does adding more layers really makes a model more efficient ?
    // A layer can be represented as a matrix A
    // But A'A (the succession of the two layers) is also a matrix
    // Maybe it is just about having enoug weights to parametrize during training?
    // I shall probably keep different matrices before training and collapse them after
    // Maybe the biases cannot be reduced with the weights
    weights: Vec<Matrix<f64>>,
    biases: Matrix<f64>,
}

impl MultiLayerPerceptron {
    fn new(architecture: Vec<u32>) -> Self {
        if architecture.len() < 2 {
            panic!("Architecture must have at least 2 layers");
        }

        let weights = Vec::new();

        // Initialize the weights
        for i in 1..architecture.len() {
            //  TODO : Set the weights random
            weights.push(Matrix::zeros(
                architecture[i - 1] as usize,
                architecture[i] as usize,
            ));
        }

        let biases = Vec::new();

        // Initialize the biases
        for i in 1..architecture.len() {
            //  TODO : Set the biases random
            biases.push(vec![0.0; architecture[i] as usize]);
        }

        Self {
            architecture,
            weights,
            biases,
        }
    }

    fn calc(&self, input: Vec<f64>) -> Vec<f64> {
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

    fn train(&mut self, database: Vec<(f64, f64)>) {}
}
