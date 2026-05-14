use super::activation_function::Function;
use super::backpropagation::backprop;
use super::io::parse_params;
use super::utils::*;

use crate::linear_algebra::addition::Add;
use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;
use crate::mlp::activation_function::Activation;

use std::cmp::min;
use std::fs::File;
use std::io::{BufReader, Write, read_to_string};
use std::path::Path;

use chrono::Local;

use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize)]
pub struct MultiLayerPerceptron {
    architecture: Vector<usize>,
    weights: Tensor<f64>,
    biases: Matrix<f64>,
    activation: Activation,
}

impl MultiLayerPerceptron {
    pub fn zeroed(architecture: Vector<usize>) -> Self {
        let (rows, columns) = into_layer(&architecture);

        let weights: Tensor<f64> = rows
            .iter()
            .zip(columns.iter())
            .map(|(r, c)| vec![vec![0.0; *c]; *r])
            .collect();

        let biases: Matrix<f64> = rows.iter().map(|r| vec![0.0; *r]).collect();

        Self {
            architecture,
            weights,
            biases,
            activation: Activation::RELU,
        }
    }
}

pub trait NeuralNetwork {
    fn new(architecture: Vector<usize>) -> Self;

    /// Forward pass : retourne uniquement la sortie finale f(z_L).
    fn calc(&self, input: Vector<f64>) -> Vector<f64>;

    /// Forward pass complet : retourne (inputs, pre_activations) pour chaque couche.
    ///
    /// - `inputs[l]`        = x_{l-1} : activation en entrée de la couche l
    /// - `pre_activations[l]` = z_l    : W_l * x_{l-1} + b_l, avant application de f
    fn forward_pass(&self, input: Vector<f64>) -> ForwardPass;

    fn backpropagation(&mut self, database: Database, iterations: usize, learning_rate: f64);

    fn gradient(&self, database: Database) -> NeuralNetGradient;

    fn error_function(&self, database: Database) -> f64;

    fn error_gradient(&self, database: Database) -> Vector<f64>;

    fn inner_gradients(&self, database: Database) -> (NeuralNetGradient, usize, f64);

    fn load(filename: &str) -> Self;
    fn save(&self) -> std::io::Result<String>;

    fn display(&self);

    fn params(&self) -> String;
    fn load_params(&mut self, file_path: &str);
}

impl NeuralNetwork for MultiLayerPerceptron {
    fn new(architecture: Vector<usize>) -> Self {
        if architecture.len() < 2 {
            panic!("Architecture must have at least 2 layers");
        }

        let (rows, columns) = into_layer(&architecture);
        let weights = (rows.clone(), columns).generate_random();
        let biases = rows.generate_random();

        Self {
            architecture,
            weights,
            biases,
            activation: Activation::RELU,
        }
    }

    fn calc(&self, input: Vector<f64>) -> Vector<f64> {
        let mut x = input;

        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let z = w.mul(&x).add(b);
            x = self.activation.apply(z);
        }

        x
    }

    fn save(&self) -> std::io::Result<String> {
        let now = Local::now();

        let filename = format!("{}.model.json", now.format("%Y-%m-%d_%H-%M-%S"));
        println!();
        print!("Writing to {}...", filename);

        let json = serde_json::to_string_pretty(self).expect("Failed to serialize model");

        let mut file = File::create(&filename)?;

        file.write_all(json.as_bytes())?;
        print!("\rWriten to {}       ", filename);
        Ok(filename)
    }

    fn load(filename: &str) -> Self {
        let file = File::open(filename).expect("Failed to open model file");

        let reader = BufReader::new(file);

        serde_json::from_reader(reader).expect("Failed to deserialize model")
    }

    fn forward_pass(&self, input: Vector<f64>) -> ForwardPass {
        let depth = self.weights.len();

        let mut inputs: Matrix<f64> = Vec::with_capacity(depth);
        let mut pre_activations: Matrix<f64> = Vec::with_capacity(depth);

        let mut x = input;

        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            // x_{l-1} : entrée de cette couche
            inputs.push(x.clone());

            // z_l = W_l * x_{l-1} + b_l
            let z = w.mul(&x).add(b);
            pre_activations.push(z.clone());

            // x_l = f(z_l)
            x = self.activation.apply(z);
        }

        ForwardPass {
            inputs,
            pre_activations,
        }
    }

    fn backpropagation(&mut self, database: Database, iterations: usize, learning_rate: f64) {
        let size = min(iterations, 100);
        for i in 0..iterations {
            let progress: usize = ((i * size) as f64 / (iterations as f64)) as usize;
            print!("\r");
            print!(
                "[{0}{1}] {i}/{iterations} iterations",
                "#".repeat(progress),
                " ".repeat(size - progress)
            );

            let (grad, _n, error) = self.inner_gradients(database.clone());

            if error <= 0.1 {
                return;
            }

            // Mise à jour : W_l -= lr * (1/n) * Σ ∂C/∂W_l
            // Les gradients dans `grad` sont déjà moyennés sur le dataset
            self.weights = self.weights.sub(&grad.weights.mul(&learning_rate));
            self.biases = self.biases.sub(&grad.biases.mul(&learning_rate));
        }
        println!();
    }

    fn gradient(&self, database: Database) -> NeuralNetGradient {
        let (grad, _, _) = self.inner_gradients(database);
        grad
    }

    /// Calcule le gradient moyen sur tout le dataset via backprop.
    fn inner_gradients(&self, database: Database) -> (NeuralNetGradient, usize, f64) {
        let n = database.len();
        let mut acc_error = 0.0;

        // Accumulateurs (même shape que weights/biases)
        let mut acc_weights: Tensor<f64> = self
            .weights
            .iter()
            .map(|w| vec![vec![0.0; w[0].len()]; w.len()])
            .collect();
        let mut acc_biases: Matrix<f64> = self.biases.iter().map(|b| vec![0.0; b.len()]).collect();

        for (input, target, coefficient) in &database {
            // 1. Forward pass : collecte z_l et x_{l-1}
            let pass = self.forward_pass(input.clone());

            // 2. Gradient de l'erreur par rapport à la sortie finale
            //    ∂C/∂x_L = 2 * (x_L - target) * coefficient
            let x_final = self
                .activation
                .apply(pass.pre_activations[pass.pre_activations.len() - 1].clone());
            acc_error += square_error(&x_final, target) * coefficient;
            let error_grad: Vector<f64> = x_final
                .iter()
                .zip(target.iter())
                .map(|(o, t)| 2.0 * (o - t) * coefficient)
                .collect();

            // 3. Backprop pour cet exemple
            let grad = backprop(
                &self.weights,
                &pass.pre_activations,
                &pass.inputs,
                error_grad,
                &self.activation,
            );

            // 4. Accumulation
            for l in 0..self.weights.len() {
                for k in 0..acc_weights[l].len() {
                    for j in 0..acc_weights[l][k].len() {
                        acc_weights[l][k][j] += grad.weights[l][k][j];
                    }
                }
                for k in 0..acc_biases[l].len() {
                    acc_biases[l][k] += grad.biases[l][k];
                }
            }
        }

        // 5. Moyenne sur le dataset
        let inv_n = 1.0 / n as f64;
        let mean_weights: Tensor<f64> = acc_weights
            .iter()
            .map(|w| {
                w.iter()
                    .map(|row| row.iter().map(|v| v * inv_n).collect())
                    .collect()
            })
            .collect();
        let mean_biases: Matrix<f64> = acc_biases
            .iter()
            .map(|b| b.iter().map(|v| v * inv_n).collect())
            .collect();

        (
            NeuralNetGradient {
                deltas: vec![], // non exposé à l'extérieur
                weights: mean_weights,
                biases: mean_biases,
            },
            n,
            acc_error,
        )
    }

    fn error_function(&self, database: Database) -> f64 {
        database
            .iter()
            .map(|(input, target, coefficient)| {
                square_error(&self.calc(input.clone()), target) * coefficient
            })
            .sum()
    }

    /// Calcule ∂C/∂x_L moyenné sur le dataset.
    /// Utilisé uniquement si on veut inspecter le gradient de sortie séparément.
    fn error_gradient(&self, database: Database) -> Vector<f64> {
        let n = database.len() as f64;
        let mut sum = vec![0.0; self.architecture[self.architecture.len() - 1]];

        for (input, target, coefficient) in &database {
            let output = self.calc(input.clone());
            for k in 0..sum.len() {
                sum[k] += 2.0 * (output[k] - target[k]) * coefficient;
            }
        }

        sum.iter().map(|v| v / n).collect()
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
        let content = read_to_string(file).unwrap();
        let (architecture, weights, biases) = parse_params(content);
        self.architecture = architecture;
        self.weights = weights;
        self.biases = biases;
    }
}
