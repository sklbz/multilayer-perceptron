use crate::mlp::activation_function::Activation;

pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn step(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub const RELU: Activation = Activation {
    function: relu,
    derivative: step,
};
