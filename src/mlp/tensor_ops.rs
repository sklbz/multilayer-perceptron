// src/mlp/tensor_ops.rs

use crate::linear_algebra::matrix::*;
use candle_core::{DType, Device, Tensor};

fn device() -> Device {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

/// Vector<f64> → Tensor candle, shape [n]
fn vec_to_tensor(v: &Vector<f64>) -> Tensor {
    Tensor::from_slice(v, (v.len(),), &device()).unwrap()
}

/// Matrix<f64> → Tensor candle, shape [rows, cols]
fn mat_to_tensor(m: &Matrix<f64>) -> Tensor {
    let rows = m.len();
    let cols = m[0].len();
    let flat: Vec<f64> = m.iter().flatten().copied().collect();
    Tensor::from_slice(&flat, (rows, cols), &device()).unwrap()
}

/// Tensor candle shape [n] → Vector<f64>
fn tensor_to_vec(t: &Tensor) -> Vector<f64> {
    t.to_dtype(DType::F64).unwrap().to_vec1().unwrap()
}

/// Tensor candle shape [rows, cols] → Matrix<f64>
fn tensor_to_mat(t: &Tensor) -> Matrix<f64> {
    t.to_dtype(DType::F64).unwrap().to_vec2().unwrap()
}

/// W · x  (forward pass)
/// W : [k, j], x : [j] → résultat : [k]
pub fn matvec(w: &Matrix<f64>, x: &Vector<f64>) -> Vector<f64> {
    let w_t = mat_to_tensor(w);
    let x_t = vec_to_tensor(x).unsqueeze(1).unwrap(); // [j, 1]
    let result = w_t.matmul(&x_t).unwrap(); // [k, 1]
    tensor_to_vec(&result.squeeze(1).unwrap()) // [k]
}

/// Wᵀ · δ  (backprop — propagation du gradient)
/// W : [k, j], δ : [k] → résultat : [j]
pub fn matvec_t(w: &Matrix<f64>, delta: &Vector<f64>) -> Vector<f64> {
    let w_t = mat_to_tensor(w).t().unwrap(); // [j, k]
    let d_t = vec_to_tensor(delta).unsqueeze(1).unwrap(); // [k, 1]
    let result = w_t.matmul(&d_t).unwrap(); // [j, 1]
    tensor_to_vec(&result.squeeze(1).unwrap()) // [j]
}

/// δ · xᵀ  (backprop — gradient des poids)
/// δ : [k], x : [j] → résultat : [k, j]
pub fn outer(delta: &Vector<f64>, x: &Vector<f64>) -> Matrix<f64> {
    let d_t = vec_to_tensor(delta).unsqueeze(1).unwrap(); // [k, 1]
    let x_t = vec_to_tensor(x).unsqueeze(0).unwrap(); // [1, j]
    let result = d_t.matmul(&x_t).unwrap(); // [k, j]
    tensor_to_mat(&result)
}
