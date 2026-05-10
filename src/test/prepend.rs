use crate::mlp::utils::Extend;

#[allow(unused)]
pub fn test() {
    let vec = vec![1f64, 2f64, 3f64];
    let result = vec.prepend(0f64);

    assert_eq!(result, vec![0., 1., 2., 3.]);
}
