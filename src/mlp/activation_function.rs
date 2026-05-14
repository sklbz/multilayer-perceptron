use crate::{linear_algebra::matrix::Vector, mlp::activation::rectified_linear::RELU};
use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize)]
pub enum Activation {
    RELU,
}

pub struct ActivationFunction {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

pub trait Function {
    fn apply(&self, input: Vector<f64>) -> Vector<f64>;
    fn gradient(&self, input: Vector<f64>) -> Vector<f64>;
}

impl Function for ActivationFunction {
    fn apply(&self, input: Vector<f64>) -> Vector<f64> {
        input
            .iter()
            .map(|coord: &f64| (self.function)(*coord))
            .collect::<Vector<f64>>()
    }

    fn gradient(&self, input: Vector<f64>) -> Vector<f64> {
        input
            .iter()
            .map(|partial: &f64| (self.derivative)(*partial))
            .collect::<Vector<f64>>()
    }
}

impl Function for Activation {
    fn apply(&self, input: Vector<f64>) -> Vector<f64> {
        match self {
            Activation::RELU => RELU.apply(input),
        }
    }
    fn gradient(&self, input: Vector<f64>) -> Vector<f64> {
        match self {
            Activation::RELU => RELU.gradient(input),
        }
    }
}
