use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::grid_display::GridDisplay;
use crate::linear_algebra::matrix::*;

pub(crate) struct MultiLayerPerceptron {
    architecture: Vector<usize>,
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

        let weights = vec![rows.clone(), colums.clone()].generate_random();

        let biases = rows.generate_random();

        Self {
            architecture,
            weights,
            biases,
        }
    }

    pub fn calc(&self, input: Vector<f64>) -> Vector<f64> {
        let result = input;

        /*
        self.weights
            .iter()
            .zip(self.biases.iter())
            .into_iter()
            .for_each(|(matrix, bias)| {
                result = matrix * result + bias;
            });
        */
        result
    }

    pub fn train(&mut self, _database: Vec<(f64, f64)>) -> () {}

    pub fn display(&self) {
        println!("Architecture:");
        println!("{:?}", self.architecture);
        println!("Weights:");
        println!("{:?}", self.weights);
        println!("Biases:");
        println!("{:?}", self.biases);
    }
}
