// src/mlp/tensor_ops.rs

use crate::linear_algebra::matrix::*;
use candle_core::{DType, Device, Tensor};
use std::sync::OnceLock;

fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| Device::cuda_if_available(0).unwrap_or(Device::Cpu))
}

/// Vector<f64> → Tensor candle, shape [n]
fn vec_to_tensor(v: &Vector<f64>) -> Tensor {
    Tensor::from_slice(v, (v.len(),), device()).unwrap()
}

/// Vector<f64> → Tensor, shape [n]  (upload biais)
pub(crate) fn vec_to_tensor_pub(v: &Vector<f64>) -> Tensor {
    vec_to_tensor(v)
}

/// z_gpu [k, N] + b_gpu [k] → [k, N]  (broadcast biais, sur GPU)
pub(crate) fn add_bias_native(z: &Tensor, b: &Tensor) -> Tensor {
    // unsqueeze b : [k] → [k, 1], puis broadcast sur N colonnes
    z.broadcast_add(&b.unsqueeze(1).unwrap()).unwrap()
}

/// Matrix<f64> → Tensor candle, shape [rows, cols]
fn mat_to_tensor(m: &Matrix<f64>) -> Tensor {
    let rows = m.len();
    let cols = m[0].len();
    let flat: Vec<f64> = m.iter().flatten().copied().collect();
    Tensor::from_slice(&flat, (rows, cols), device()).unwrap()
}

/// Tensor candle shape [n] → Vector<f64>
fn tensor_to_vec(t: &Tensor) -> Vector<f64> {
    t.to_dtype(DType::F64).unwrap().to_vec1().unwrap()
}

pub(crate) fn tensor_to_vec_pub(t: &Tensor) -> Vector<f64> {
    tensor_to_vec(t)
}

pub(crate) fn row_means_tensor(m: &Tensor) -> Tensor {
    m.mean(1).unwrap()
}

/// Tensor candle shape [rows, cols] → Matrix<f64>
fn tensor_to_mat(t: &Tensor) -> Matrix<f64> {
    t.to_dtype(DType::F64).unwrap().to_vec2().unwrap()
}

/// W ← W - dW, entièrement sur GPU
pub(crate) fn tensor_sub_native(w: &Tensor, dw: &Tensor) -> Tensor {
    (w - dw).unwrap()
}
/// Tenseur de zéros de même shape que `t`, sur GPU
pub(crate) fn zeros_like_gpu(t: &Tensor) -> Tensor {
    Tensor::zeros_like(t).unwrap()
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
/// W · X  (forward pass batché)
/// W : [k, j], X : [j, N] → résultat : [k, N]
pub fn matmul(w: &Matrix<f64>, x: &Matrix<f64>) -> Matrix<f64> {
    let w_t = mat_to_tensor(w);
    let x_t = mat_to_tensor(x);
    let result = w_t.matmul(&x_t).unwrap();
    tensor_to_mat(&result)
}

/// Wᵀ · D  (backprop batché — propagation du gradient)
/// W : [k, j], D : [k, N] → résultat : [j, N]
pub fn matmul_t(w: &Matrix<f64>, delta: &Matrix<f64>) -> Matrix<f64> {
    let w_t = mat_to_tensor(w).t().unwrap();
    let d_t = mat_to_tensor(delta);
    let result = w_t.matmul(&d_t).unwrap();
    tensor_to_mat(&result)
}

/// D · Xᵀ  (backprop batché — gradient des poids, moyenné sur le batch)
/// D : [k, N], X : [j, N] → résultat : [k, j]
pub fn outer_batch(delta: &Matrix<f64>, x: &Matrix<f64>) -> Matrix<f64> {
    let n = delta[0].len() as f64;
    let d_t = mat_to_tensor(delta); // [k, N]
    let x_t = mat_to_tensor(x).t().unwrap(); // [N, j]
    let result = d_t.matmul(&x_t).unwrap(); // [k, j]
    // Moyenne sur le batch
    tensor_to_mat(&result.affine(1.0 / n, 0.0).unwrap())
}

/// W · X sans round-trip — entrée et sortie restent sur GPU
pub(crate) fn matmul_native(w: &Tensor, x: &Tensor) -> Tensor {
    w.matmul(x).unwrap()
}

/// Wᵀ · D sans round-trip
pub(crate) fn matmul_t_native(w: &Tensor, x: &Tensor) -> Tensor {
    w.t().unwrap().matmul(x).unwrap()
}

/// D · Xᵀ moyenné sur le batch, sans round-trip
pub(crate) fn outer_batch_native(delta: &Tensor, x: &Tensor, n: f64) -> Tensor {
    let x_t = x.t().unwrap();
    delta.matmul(&x_t).unwrap().affine(1.0 / n, 0.0).unwrap()
}

/// Matrix<f64> → Tensor (upload unique en début de step)
pub(crate) fn mat_to_tensor_pub(m: &Matrix<f64>) -> Tensor {
    mat_to_tensor(m)
}

/// Tensor → Matrix<f64> (download unique en fin de step)
pub(crate) fn tensor_to_mat_pub(t: &Tensor) -> Matrix<f64> {
    tensor_to_mat(t)
}
