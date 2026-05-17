use super::activation_function::Function;
use super::io::parse_params;
use super::utils::*;

use crate::linear_algebra::addition::Add;
use crate::linear_algebra::generator::Generator;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;
use crate::mlp::activation_function::Activation;
use crate::mlp::backpropagation::backprop_batch;
use crate::mlp::tensor_ops::matmul;

use core::f64;
use std::cmp::min;
use std::fs::File;
use std::io::{BufReader, Write, read_to_string, stdout};
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
    fn calc(&self, input: &[f64]) -> Vector<f64>;

    /// Forward pass complet : retourne (inputs, pre_activations) pour chaque couche.
    ///
    /// - `inputs[l]`        = x_{l-1} : activation en entrée de la couche l
    /// - `pre_activations[l]` = z_l    : W_l * x_{l-1} + b_l, avant application de f
    fn forward_pass(&self, input_batch: &Matrix<f64>) -> ForwardPass;

    fn backpropagation(&mut self, database: &Database, iterations: usize, learning_rate: f64);

    fn gradient(&self, database: &Database) -> NeuralNetGradient;

    fn error_function(&self, database: &Database) -> f64;

    fn error_gradient(&self, database: &Database) -> Vector<f64>;

    fn inner_gradients(&self, database: &Database) -> (NeuralNetGradient, usize, f64);

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

    fn calc(&self, input: &[f64]) -> Vector<f64> {
        let mut x = input.to_vec();

        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let z = w.mul(&x).add(b);
            x = self.activation.apply(z);
        }

        x
    }

    fn save(&self) -> std::io::Result<String> {
        let now = Local::now();

        let filename = format!("{}.model.json", now.format("%Y-%m-%d_%H-%M-%S"));
        print!("Writing to {}...", filename);

        let json = serde_json::to_string_pretty(self).expect("Failed to serialize model");

        let mut file = File::create(&filename)?;

        file.write_all(json.as_bytes())?;
        print!("\rWriten to {}  ", filename);
        Ok(filename)
    }

    fn load(filename: &str) -> Self {
        let file = File::open(filename).expect("Failed to open model file");

        let reader = BufReader::new(file);

        serde_json::from_reader(reader).expect("Failed to deserialize model")
    }

    fn forward_pass(&self, inputs_batch: &Matrix<f64>) -> ForwardPass {
        let depth = self.weights.len();
        let mut inputs: Vec<Matrix<f64>> = Vec::with_capacity(depth);
        let mut pre_activations: Vec<Matrix<f64>> = Vec::with_capacity(depth);

        let mut x = inputs_batch.clone(); // [j, N]

        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            // z_l = W_l · X + b_l  — b est broadcasté sur les N colonnes
            let mut z = matmul(w, &x); // [k, N]
            for col in z.iter_mut() {
                for (val, bias) in col.iter_mut().zip(b.iter()) {
                    *val += bias;
                }
            }

            let x_next: Matrix<f64> = z
                .iter()
                .map(|col| self.activation.apply(col.clone()))
                .collect();
            pre_activations.push(z.clone());

            // Move de x et z : plus aucun clone nécessaire
            inputs.push(x);
            pre_activations.push(z);

            x = x_next;
        }

        ForwardPass {
            inputs,
            pre_activations,
        }
    }

    fn backpropagation(&mut self, database: &Database, iterations: usize, learning_rate: f64) {
        let mut adam = AdamState::new(&self.weights, &self.biases);
        let size = min(iterations, 75);
        let mut min_err = f64::INFINITY;
        let mut initial_err: Option<f64> = None;
        for i in 0..iterations {
            let progress: usize = ((i * size + 1) as f64 / (iterations as f64)) as usize;
            print!(
                "\r[{0}{1}] {2}/{iterations} ",
                "#".repeat(progress),
                " ".repeat(size - progress - 1),
                i + 1,
            );

            let (grad, _n, error) = self.inner_gradients(database);
            if min_err > error {
                min_err = min_err.min(error);
            }

            if initial_err.is_none() {
                initial_err = Some(error);
            }

            print!("err: {:.2e}, min: {:.2e} ", error, min_err);
            if let Some(initial) = initial_err {
                print!("initial: {:.2e}", initial)
            };
            stdout().flush().unwrap();

            if error <= 0.1 {
                return;
            }

            let (dw, db) = adam_update(
                &mut adam,
                &grad.weights,
                &grad.biases,
                learning_rate,
                0.9,
                0.999,
                1e-8,
            );
            self.weights = self.weights.sub(&dw);
            self.biases = self.biases.sub(&db);
        }
        print!("\r");
    }

    fn gradient(&self, database: &Database) -> NeuralNetGradient {
        let (grad, _, _) = self.inner_gradients(database);
        grad
    }

    /// Calcule le gradient moyen sur tout le dataset via backprop.
    fn inner_gradients(&self, database: &Database) -> (NeuralNetGradient, usize, f64) {
        let n = database.len();

        // 1. Construire la matrice d'entrée X : shape [n_features, N]
        //    et la matrice de cibles T : shape [n_outputs, N]
        let inputs_batch: Matrix<f64> = transpose(
            &database
                .iter()
                .map(|(x, _, _)| x.as_ref().to_vec())
                .collect(),
        ); // [768, N]

        let targets_batch: Matrix<f64> = transpose(
            &database
                .iter()
                .map(|(_, t, _)| t.as_ref().to_vec())
                .collect(),
        ); // [n_outputs, N]

        let coefficients: Vector<f64> = database.iter().map(|(_, _, c)| *c).collect();

        // 2. Forward pass batché — un seul appel GPU
        let pass = self.forward_pass(&inputs_batch);

        // 3. Sortie finale x_L : shape [n_outputs, N]
        let last_pre_act = pass.pre_activations.last().unwrap();
        let x_final: Matrix<f64> = last_pre_act
            .iter()
            .map(|col| self.activation.apply(col.clone()))
            .collect();

        // 4. Erreur scalaire : Σ coefficient_i * ||x_L_i - t_i||²
        let acc_error: f64 = (0..n)
            .map(|i| {
                let col: Vector<f64> = x_final.iter().map(|row| row[i]).collect();
                let target: Vector<f64> = targets_batch.iter().map(|row| row[i]).collect();
                square_error(&col, &target) * coefficients[i]
            })
            .sum();

        // 5. Gradient de l'erreur : shape [n_outputs, N]
        //    ∂C/∂x_L[:, i] = 2 * (x_L[:, i] - t[:, i]) * coefficient_i
        let error_grad_batch: Matrix<f64> = x_final
            .iter()
            .zip(targets_batch.iter())
            .map(|(out_row, tgt_row)| {
                out_row
                    .iter()
                    .zip(tgt_row.iter())
                    .zip(coefficients.iter())
                    .map(|((o, t), c)| 2.0 * (o - t) * c)
                    .collect()
            })
            .collect();

        let grad = backprop_batch(
            &self.weights,
            &pass.pre_activations,
            &pass.inputs,
            error_grad_batch,
            &self.activation,
        );

        (
            NeuralNetGradient {
                weights: grad.weights,
                biases: grad.biases,
            },
            n,
            acc_error,
        )
    }

    fn error_function(&self, database: &Database) -> f64 {
        database
            .iter()
            .map(|(input, target, coefficient)| {
                square_error(&self.calc(input), target) * coefficient
            })
            .sum()
    }

    /// Calcule ∂C/∂x_L moyenné sur le dataset.
    /// Utilisé uniquement si on veut inspecter le gradient de sortie séparément.
    fn error_gradient(&self, database: &Database) -> Vector<f64> {
        let n = database.len() as f64;
        let mut sum = vec![0.0; self.architecture[self.architecture.len() - 1]];

        for (input, target, coefficient) in database {
            let output = self.calc(input);
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
