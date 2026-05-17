use super::utils::*;
use crate::linear_algebra::matrix::Matrix;
use crate::linear_algebra::matrix::Vector;
use crate::mlp::activation_function::Activation;
use crate::mlp::activation_function::Function;
use crate::mlp::tensor_ops::row_means_tensor;
use crate::mlp::tensor_ops::tensor_to_vec_pub;
use candle_core::Tensor;

/// Version batché de backprop.
///
/// # Arguments
/// - `weights`           : W_l pour chaque couche, shape [L][k][j]
/// - `pre_acts`          : z_l pour chaque couche, shape [L][k][N]
/// - `inputs`            : x_{l-1} pour chaque couche, shape [L][j][N]
/// - `error_grad_batch`  : ∂C/∂x_L, shape [n_outputs][N]
/// - `activation`        : fonction d'activation
pub(super) fn backprop_batch(
    weights: &[Matrix<f64>], // types inchangés côté appelant
    pre_acts: &[Matrix<f64>],
    inputs: &[Matrix<f64>],
    error_grad_batch: &Matrix<f64>,
    activation: &Activation,
) -> NeuralNetGradient {
    use crate::mlp::tensor_ops::{
        mat_to_tensor_pub as upload, matmul_t_native, outer_batch_native,
        tensor_to_mat_pub as download,
    };

    let depth = weights.len();
    let n = error_grad_batch[0].len() as f64;

    // ── Upload unique ──────────────────────────────────────────────
    // weights déjà en RAM : un upload par couche, fait une seule fois
    let w_gpu: Vec<Tensor> = weights.iter().map(upload).collect();
    let inputs_gpu: Vec<Tensor> = inputs.iter().map(upload).collect();
    let pre_acts_gpu: Vec<Tensor> = pre_acts.iter().map(upload).collect();
    let err_gpu = upload(error_grad_batch);

    // ── Couche L ───────────────────────────────────────────────────
    let f_prime_last = activation.gradient_tensor(&pre_acts_gpu[depth - 1]);
    let delta_last = (err_gpu * f_prime_last).unwrap(); // hadamard sur GPU

    let mut weight_grads_gpu: Vec<Tensor> = Vec::with_capacity(depth);
    let mut bias_grads_gpu: Vec<Tensor> = Vec::with_capacity(depth);
    let mut deltas_gpu: Vec<Tensor> = Vec::with_capacity(depth);

    weight_grads_gpu.push(outer_batch_native(&delta_last, &inputs_gpu[depth - 1], n));
    bias_grads_gpu.push(row_means_tensor(&delta_last));
    deltas_gpu.push(delta_last);

    // ── Couches L-1 → 0, entièrement sur GPU ──────────────────────
    for l in (0..depth - 1).rev() {
        let delta_next = deltas_gpu.last().unwrap();
        let propagated = matmul_t_native(&w_gpu[l + 1], delta_next);

        let f_prime = activation.gradient_tensor(&pre_acts_gpu[l]);
        let delta_l = (propagated * f_prime).unwrap();

        weight_grads_gpu.push(outer_batch_native(&delta_l, &inputs_gpu[l], n));
        bias_grads_gpu.push(row_means_tensor(&delta_l));
        deltas_gpu.push(delta_l);
    }

    weight_grads_gpu.reverse();
    bias_grads_gpu.reverse();

    // ── Download unique ────────────────────────────────────────────
    NeuralNetGradient {
        weights: weight_grads_gpu.iter().map(download).collect(),
        biases: bias_grads_gpu
            .iter()
            .map(|t| {
                tensor_to_vec_pub(t) // row_means produit déjà [k]
            })
            .collect(),
    }
}

/// Hadamard batch : A ⊙ B, shape [k][N] ⊙ [k][N] → [k][N]
fn hadamard_batch(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(row_a, row_b)| hadamard(row_a, row_b))
        .collect()
}

/// Moyenne de chaque ligne sur N exemples : [k][N] → [k]
fn row_means(m: &Matrix<f64>) -> Vector<f64> {
    let n = m[0].len() as f64;
    m.iter().map(|row| row.iter().sum::<f64>() / n).collect()
}

/// Produit de Hadamard (élément par élément) entre deux vecteurs.
fn hadamard(a: &Vector<f64>, b: &Vector<f64>) -> Vector<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "Hadamard: vecteurs de tailles différentes ({} vs {})",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}
