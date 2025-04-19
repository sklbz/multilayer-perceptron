use super::utils::*;
use crate::linear_algebra::matrix::*;
use crate::linear_algebra::product::*;

//TODO: test all this mess
pub(super) fn backprop(
    weight_partials: Tensor<f64>,
    activation_partials: Tensor<f64>,
    grad: NeuralNetGradient,
    depth: usize,
) -> NeuralNetGradient {
    if depth == 0 {
        return grad;
    }

    let weight: &Matrix<f64> = &weight_partials[depth - 1];
    let activation: &Matrix<f64> = &activation_partials[depth - 1];

    let previous = previous_layer_gradient(activation, weight, grad.neurons[0].clone());

    backprop(
        weight_partials,
        activation_partials,
        extend_gradient(grad, previous),
        depth - 1,
    )
}

//----------------------------------------------------------------------------------------------

pub(super) fn previous_layer_gradient(
    activation: &Matrix<f64>,
    weight: &Matrix<f64>,
    neurons: Vector<f64>,
) -> GradientLayer {
    // DEBUG--------------------------------------------------------------------------------------
    println!(
        "---------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    println!("DEBUG");
    println!();
    fn size(name: String, matrix: &Matrix<f64>) {
        println!("{name} size: {0}x{1}", matrix.len(), matrix[0].len());
        println!("{name}: {:?}", matrix);
    }
    println!("Layer size: {}", neurons.len());
    size("activation".to_string(), activation);
    size("weight".to_string(), weight);
    println!();
    println!(
        "---------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    // --------------------------------------------------------------------------------------

    println!(">>>>>");
    //                       ∂ cost
    // previous_layer[k] = -----------
    //                     ∂(neuron k)
    let previous_layer: Vector<f64> = activation.transpose().mul(&neurons.clone());
    println!("<<<<<");

    //                             ∂ cost
    // previous_weights[k, j] = ------------
    //                          ∂(weight kj)
    let previous_weights: Matrix<f64> = weight
        .iter()
        .zip(neurons.iter())
        .map(|(weight_partial, neuron_partial): (&Vector<f64>, &f64)| {
            weight_partial.mul(neuron_partial)
        })
        .collect::<Matrix<f64>>();

    //                       ∂ cost
    // previous_biases[k] = ---------
    //                      ∂(bias k)
    let previous_biases = neurons.clone();

    GradientLayer {
        neurons: previous_layer,
        weights: previous_weights,
        biases: previous_biases,
    }
}

//----------------------------------------------------------------------------------------------

pub(super) fn extend_gradient(grad: NeuralNetGradient, layer: GradientLayer) -> NeuralNetGradient {
    NeuralNetGradient {
        neurons: grad.neurons.prepend(layer.neurons),
        weights: grad.weights.prepend(layer.weights),
        biases: grad.biases.prepend(layer.biases),
    }
}

//----------------------------------------------------------------------------------------------

pub(super) fn extract_last_layer(results: Gradient<Vector<f64>>) -> NeuralNetGradient {
    NeuralNetGradient {
        neurons: vec![results],
        weights: vec![],
        biases: vec![],
    }
}
