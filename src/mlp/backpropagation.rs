use super::utils::*;
use crate::linear_algebra::matrix::*;
use crate::mlp::activation_function::Activation;
use crate::mlp::activation_function::Function;
use crate::mlp::tensor_ops::{matvec_t, outer};

use crate::mlp::tensor_ops::{matmul_t, outer_batch};

/// Version batché de backprop.
///
/// # Arguments
/// - `weights`           : W_l pour chaque couche, shape [L][k][j]
/// - `pre_acts`          : z_l pour chaque couche, shape [L][k][N]
/// - `inputs`            : x_{l-1} pour chaque couche, shape [L][j][N]
/// - `error_grad_batch`  : ∂C/∂x_L, shape [n_outputs][N]
/// - `activation`        : fonction d'activation
pub(super) fn backprop_batch(
    weights: &Tensor<f64>,
    pre_acts: &[Matrix<f64>],
    inputs: &[Matrix<f64>],
    error_grad_batch: Matrix<f64>,
    activation: &Activation,
) -> NeuralNetGradient {
    let depth = weights.len();

    // δ_L = ∇_a(C) ⊙ f'(z_L)  — shape [k, N]
    let f_prime_last: Matrix<f64> = pre_acts[depth - 1]
        .iter()
        .map(|col| activation.gradient(col.clone()))
        .collect();
    let delta_last: Matrix<f64> = hadamard_batch(&error_grad_batch, &f_prime_last);

    // ∂C/∂W_L = δ_L · x_{L-1}ᵀ  moyenné sur N — shape [k, j]
    let mut weight_grads: Tensor<f64> = Vec::with_capacity(depth);
    // ∂C/∂b_L = mean(δ_L, axis=N) — shape [k]
    let mut bias_grads: Matrix<f64> = Vec::with_capacity(depth);
    let mut deltas: Vec<Matrix<f64>> = Vec::with_capacity(depth);

    weight_grads.push(outer_batch(&delta_last, &inputs[depth - 1]));
    bias_grads.push(row_means(&delta_last));
    deltas.push(delta_last);

    // Rétropropagation de L-1 jusqu'à 0
    for l in (0..depth - 1).rev() {
        let delta_next: &Vec<Vec<f64>> = deltas.last().unwrap();

        let w_next = &weights[l + 1];

        // W_{l+1}ᵀ · δ_{l+1} — shape [j, N]
        let propagated = matmul_t(w_next, delta_next);

        // ⊙ f'(z_l) — shape [j, N]
        let f_prime: Matrix<f64> = pre_acts[l]
            .iter()
            .map(|col| activation.gradient(col.clone()))
            .collect();
        let delta_l = hadamard_batch(&propagated, &f_prime);

        // ∂C/∂W_l = δ_l · x_{l-1}ᵀ  moyenné sur N
        weight_grads.push(outer_batch(&delta_l, &inputs[l]));
        // ∂C/∂b_l = mean(δ_l, axis=N)
        bias_grads.push(row_means(&delta_l));
        deltas.push(delta_l);
    }
    weight_grads.reverse();
    bias_grads.reverse();

    NeuralNetGradient {
        weights: weight_grads,
        biases: bias_grads,
    }
}

// ------------------------------------------------------------------------------------

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

/// Point d'entrée de la backpropagation.
///
/// # Arguments
/// - `weights`      : W_l pour chaque couche l, shape [L][k][j]
/// - `pre_acts`     : z_l = W_l * x_{l-1} + b_l pour chaque couche l, shape [L][k]
/// - `inputs`       : x_{l-1} = f(z_{l-1}) en entrée de chaque couche, shape [L][j]
/// - `error_grad`   : ∇_a(C) = ∂C/∂x_L, gradient de l'erreur par rapport à la sortie finale
/// - `activation`   : fonction d'activation (pour sa dérivée f')
///
/// # Retourne
/// `NeuralNetGradient` contenant δ_l, ∂C/∂W_l et ∂C/∂b_l pour chaque couche.
pub fn backprop(
    weights: &Tensor<f64>,
    pre_acts: &Matrix<f64>,
    inputs: &Matrix<f64>,
    error_grad: Vector<f64>,
    activation: &Activation,
) -> NeuralNetGradient {
    let depth = weights.len(); // nombre de couches

    // δ_L = ∇_a(C) ⊙ f'(z_L)
    let f_prime_last = activation.gradient(pre_acts[depth - 1].clone());
    let delta_last = hadamard(&error_grad, &f_prime_last);

    // Initialisation avec la dernière couche
    let mut deltas: Matrix<f64> = vec![delta_last.clone()];
    let mut weight_grads: Tensor<f64> = vec![outer(&delta_last, &inputs[depth - 1])];
    let mut bias_grads: Matrix<f64> = vec![delta_last];

    // Rétropropagation couche par couche, de L-1 jusqu'à 0
    for l in (0..depth - 1).rev() {
        // δ_l = (W_{l+1}ᵀ · δ_{l+1}) ⊙ f'(z_l)
        let delta_next = &deltas[0]; // deltas est en ordre inverse, [0] = couche l+1
        let w_next = &weights[l + 1]; // W_{l+1}

        // W_{l+1}ᵀ · δ_{l+1}
        let propagated: Vector<f64> = matvec_t(w_next, delta_next);

        // ⊙ f'(z_l)
        let f_prime = activation.gradient(pre_acts[l].clone());
        let delta_l = hadamard(&propagated, &f_prime);

        // ∂C/∂W_l = δ_l · x_{l-1}ᵀ  (produit externe)
        let weight_grad_l = outer(&delta_l, &inputs[l]);

        // ∂C/∂b_l = δ_l
        let bias_grad_l = delta_l.clone();

        // Prepend pour garder l'ordre [couche 0, ..., couche L]
        deltas = deltas.prepend(delta_l);
        weight_grads = weight_grads.prepend(weight_grad_l);
        bias_grads = bias_grads.prepend(bias_grad_l);
    }

    NeuralNetGradient {
        weights: weight_grads,
        biases: bias_grads,
    }
}
