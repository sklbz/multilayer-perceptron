use crate::mlp::multilayer_perceptron::*;

#[allow(unused)]
use std::thread::sleep;
#[allow(unused)]
use std::time::Duration;

#[allow(unused)]
pub fn test() {
    let mut mlp = MultiLayerPerceptron::new(vec![2, 3, 4, 1]);

    let database = vec![
        (vec![0f64, 0f64], vec![0f64], 1f64),
        (vec![0f64, 1f64], vec![0f64], 1f64),
        (vec![1f64, 0f64], vec![0f64], 1f64),
        (vec![1f64, 1f64], vec![0f64], 1f64),
        (vec![-1f64, 0f64], vec![0f64], 1f64),
        (vec![-1f64, 1f64], vec![0f64], 1f64),
        (vec![0f64, -1f64], vec![0f64], 1f64),
        (vec![1f64, -1f64], vec![0f64], 1f64),
        (vec![-1f64, -1f64], vec![0f64], 1f64),
    ];

    mlp.backpropagation(database.clone());

    sleep(Duration::from_millis(1000));

    let test = mlp.calc(vec![0f64, 0f64]);

    println!();
    println!("TEST: {:?}", test);
}
