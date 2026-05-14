use super::utils::*;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::*;
use crate::mlp::activation_function::Activation;
use crate::mlp::activation_function::Function;

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
pub(super) fn backprop(
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
    let mut weight_grads: Tensor<f64> = vec![outer_product(&delta_last, &inputs[depth - 1])];
    let mut bias_grads: Matrix<f64> = vec![delta_last];

    // Rétropropagation couche par couche, de L-1 jusqu'à 0
    for l in (0..depth - 1).rev() {
        // δ_l = (W_{l+1}ᵀ · δ_{l+1}) ⊙ f'(z_l)
        let delta_next = &deltas[0]; // deltas est en ordre inverse, [0] = couche l+1
        let w_next = &weights[l + 1]; // W_{l+1}

        // W_{l+1}ᵀ · δ_{l+1}
        let propagated: Vector<f64> = w_next.transpose().mul(delta_next);

        // ⊙ f'(z_l)
        let f_prime = activation.gradient(pre_acts[l].clone());
        let delta_l = hadamard(&propagated, &f_prime);

        // ∂C/∂W_l = δ_l · x_{l-1}ᵀ  (produit externe)
        let weight_grad_l = outer_product(&delta_l, &inputs[l]);

        // ∂C/∂b_l = δ_l
        let bias_grad_l = delta_l.clone();

        // Prepend pour garder l'ordre [couche 0, ..., couche L]
        deltas = deltas.prepend(delta_l);
        weight_grads = weight_grads.prepend(weight_grad_l);
        bias_grads = bias_grads.prepend(bias_grad_l);
    }

    NeuralNetGradient {
        deltas,
        weights: weight_grads,
        biases: bias_grads,
    }
}

// ------------------------------------------------------------------------------------

/// Produit externe : u · vᵀ, retourne une matrice [len(u)][len(v)].
///
/// ∂C/∂W_l[k, j] = δ_l[k] * x_{l-1}[j]
fn outer_product(u: &Vector<f64>, v: &Vector<f64>) -> Matrix<f64> {
    u.iter()
        .map(|ui| v.iter().map(|vj| ui * vj).collect())
        .collect()
}
