use crate::{linear_algebra::matrix::Vector, mlp::activation::rectified_linear::RELU};
use candle_core::DType;
use candle_core::Tensor;
use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize)]
pub enum Activation {
    ReLU,
}

pub struct ActivationFunction {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

pub trait Function {
    fn apply(&self, input: &[f64]) -> Vector<f64>;
    fn apply_tensor(&self, input: &Tensor) -> Tensor;
    fn gradient(&self, input: &[f64]) -> Vector<f64>;
    fn gradient_tensor(&self, pre_act: &Tensor) -> Tensor;
}

impl Function for ActivationFunction {
    fn apply(&self, input: &[f64]) -> Vector<f64> {
        input
            .iter()
            .map(|coord: &f64| (self.function)(*coord))
            .collect::<Vector<f64>>()
    }
    fn apply_tensor(&self, input: &Tensor) -> Tensor {
        let v: Vec<f64> = input
            .to_dtype(DType::F64)
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let applied: Vec<f64> = v.iter().map(|x| (self.function)(*x)).collect();
        Tensor::from_slice(&applied, input.shape(), input.device()).unwrap()
    }
    fn gradient(&self, input: &[f64]) -> Vector<f64> {
        input
            .iter()
            .map(|partial: &f64| (self.derivative)(*partial))
            .collect::<Vector<f64>>()
    }
    fn gradient_tensor(&self, pre_act: &Tensor) -> Tensor {
        let v: Vec<f64> = pre_act
            .to_dtype(DType::F64)
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let derived: Vec<f64> = v.iter().map(|x| (self.derivative)(*x)).collect();
        Tensor::from_slice(&derived, pre_act.shape(), pre_act.device()).unwrap()
    }
}

impl Function for Activation {
    fn apply(&self, input: &[f64]) -> Vector<f64> {
        match self {
            Activation::ReLU => RELU.apply(input),
        }
    }
    fn apply_tensor(&self, input: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => {
                // leaky_relu(x) = relu(x) - 0.05 * relu(-x)
                let pos = input.relu().unwrap();
                let neg = (input.neg().unwrap().relu().unwrap() * 0.05f64).unwrap();
                (pos - neg).unwrap()
            }
        }
    }
    fn gradient(&self, input: &[f64]) -> Vector<f64> {
        match self {
            Activation::ReLU => RELU.gradient(input),
        }
    }
    fn gradient_tensor(&self, pre_act: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => {
                let mask = pre_act.gt(0.0f64).unwrap().to_dtype(DType::F64).unwrap();
                let mask_neg = (1.0f64 - &mask).unwrap();
                (mask + (mask_neg * 0.05f64).unwrap()).unwrap()
            }
        }
    }
}
