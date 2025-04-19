use crate::mlp::utils::Extend;

#[allow(unused)]
pub fn test() {
    let vec = vec![1f64, 2f64, 3f64];

    println!("{:?}", vec);

    let result = vec.prepend(0f64);

    println!("{:?}", result);
}
