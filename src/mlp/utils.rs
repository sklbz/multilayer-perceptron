use std::sync::Arc;

use crate::linear_algebra::addition::Add;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::Mul;

pub fn into_layer(architecture: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let layers_count = architecture.len() - 1;
    let rows: Vec<usize> = architecture.get(1..).unwrap_or(&[]).to_vec();
    let columns: Vec<usize> = architecture.get(..layers_count).unwrap_or(&[]).to_vec();
    (rows, columns)
}

pub fn square_error(result: &[f64], target: &[f64]) -> f64 {
    result
        .iter()
        .zip(target.iter())
        .map(|(calc, target)| (calc - target).powf(2.0))
        .sum::<f64>()
}

// ------------------------------------------------------------------------------------

pub(super) trait Mean {
    type Output;
    fn mean(&self) -> Self::Output;
}

impl Mean for Tensor<f64> {
    type Output = Matrix<f64>;

    fn mean(&self) -> Self::Output {
        let mut mean = self[0].clone().mul(&0f64);
        self.iter().for_each(|matrix| mean = mean.add(matrix));
        let multiplier = 1f64 / self.len() as f64;
        mean.mul(&multiplier)
    }
}

impl Mean for Matrix<f64> {
    type Output = Vector<f64>;

    fn mean(&self) -> Self::Output {
        let mut mean = vec![0f64; self[0].len()];
        self.iter().for_each(|vector| mean = mean.add(vector));
        let multiplier = 1f64 / self.len() as f64;
        mean.mul(&multiplier)
    }
}

impl Mean for Vector<f64> {
    type Output = f64;

    fn mean(&self) -> Self::Output {
        let multiplier = 1f64 / self.len() as f64;
        self.iter().sum::<f64>() * multiplier
    }
}

// ------------------------------------------------------------------------------------
/// Transpose une Matrix<f64> : [N][k] → [k][N]
pub fn transpose(m: &Matrix<f64>) -> Matrix<f64> {
    if m.is_empty() {
        return vec![];
    }
    let rows = m.len();
    let cols = m[0].len();
    (0..cols)
        .map(|j| (0..rows).map(|i| m[i][j]).collect())
        .collect()
}
// ------------------------------------------------------------------------------------

pub fn shared(v: Vec<f64>) -> SharedVector {
    Arc::from(v)
}

pub type SharedVector = Arc<[f64]>;

pub type Database = Vec<(SharedVector, SharedVector, f64)>;

/// Résultat du forward pass pour un batch donné.
pub struct ForwardPass {
    pub inputs: Vec<Matrix<f64>>,          // [L][j, N]
    pub pre_activations: Vec<Matrix<f64>>, // [L][k, N]
}

pub struct StepwiseGradients {
    pub forward_passes: Vec<ForwardPass>,
    pub results: Vector<f64>,
}

// ------------------------------------------------------------------------------------

pub trait Extend<T> {
    fn append(self, value: T) -> Self;
    fn prepend(self, value: T) -> Self;
}

impl<T> Extend<T> for Vec<T> {
    fn append(mut self, value: T) -> Self {
        self.splice(self.len()..self.len(), vec![value]);
        self
    }

    fn prepend(mut self, value: T) -> Self {
        self.splice(0..0, vec![value]);
        self
    }
}

// ------------------------------------------------------------------------------------

pub type Gradient<T> = T;

/// Gradient complet du réseau, couche par couche.
pub struct NeuralNetGradient {
    /// ∂C/∂W_l pour chaque couche
    pub weights: Gradient<Tensor<f64>>,
    /// ∂C/∂b_l = δ_l pour chaque couche
    pub biases: Gradient<Matrix<f64>>,
}

// ------------------------------------------------------------------------------------
/// État interne de l'optimiseur Adam.
pub struct AdamState {
    pub m_weights: Tensor<f64>, // moments d'ordre 1 des poids
    pub v_weights: Tensor<f64>, // moments d'ordre 2 des poids
    pub m_biases: Matrix<f64>,
    pub v_biases: Matrix<f64>,
    pub t: usize, // compteur d'itérations
}

impl AdamState {
    /// Initialise tous les moments à zéro, même shape que le réseau.
    pub fn new(weights: &Tensor<f64>, biases: &Matrix<f64>) -> Self {
        let zero_w = |w: &Matrix<f64>| vec![vec![0.0; w[0].len()]; w.len()];
        let zero_b = |b: &Vector<f64>| vec![0.0; b.len()];
        Self {
            m_weights: weights.iter().map(zero_w).collect(),
            v_weights: weights.iter().map(zero_w).collect(),
            m_biases: biases.iter().map(zero_b).collect(),
            v_biases: biases.iter().map(zero_b).collect(),
            t: 0,
        }
    }
}

pub fn adam_update(
    state: &mut AdamState,
    grad_weights: &Tensor<f64>,
    grad_biases: &Matrix<f64>,
    lr: f64,
    beta1: f64, // 0.9
    beta2: f64, // 0.999
    eps: f64,   // 1e-8
) -> (Tensor<f64>, Matrix<f64>) {
    state.t += 1;
    let t = state.t as f64;

    let bc1 = 1.0 - beta1.powf(t); // correction biais ordre 1
    let bc2 = 1.0 - beta2.powf(t); // correction biais ordre 2

    let mut dw: Tensor<f64> = Vec::new();
    let mut db: Matrix<f64> = Vec::new();

    for l in 0..grad_weights.len() {
        let mut dw_l: Matrix<f64> = Vec::new();
        for k in 0..grad_weights[l].len() {
            let mut dw_lk: Vector<f64> = Vec::new();
            for j in 0..grad_weights[l][k].len() {
                let g = grad_weights[l][k][j];
                state.m_weights[l][k][j] = beta1 * state.m_weights[l][k][j] + (1.0 - beta1) * g;
                state.v_weights[l][k][j] = beta2 * state.v_weights[l][k][j] + (1.0 - beta2) * g * g;
                let m_hat = state.m_weights[l][k][j] / bc1;
                let v_hat = state.v_weights[l][k][j] / bc2;
                dw_lk.push(lr * m_hat / (v_hat.sqrt() + eps));
            }
            dw_l.push(dw_lk);
        }
        dw.push(dw_l);

        let mut db_l: Vector<f64> = Vec::new();
        for k in 0..grad_biases[l].len() {
            let g = grad_biases[l][k];
            state.m_biases[l][k] = beta1 * state.m_biases[l][k] + (1.0 - beta1) * g;
            state.v_biases[l][k] = beta2 * state.v_biases[l][k] + (1.0 - beta2) * g * g;
            let m_hat = state.m_biases[l][k] / bc1;
            let v_hat = state.v_biases[l][k] / bc2;
            db_l.push(lr * m_hat / (v_hat.sqrt() + eps));
        }
        db.push(db_l);
    }

    (dw, db)
}
