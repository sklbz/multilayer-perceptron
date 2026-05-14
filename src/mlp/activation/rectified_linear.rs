use crate::mlp::activation_function::ActivationFunction;

pub fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.05 * x }
}

pub fn step(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.05 }
}

pub const RELU: ActivationFunction = ActivationFunction {
    function: leaky_relu,
    derivative: step,
};
