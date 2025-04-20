use crate::mlp::multilayer_perceptron::*;

#[allow(unused)]
use std::thread::sleep;
#[allow(unused)]
use std::time::Duration;

#[allow(unused)]
pub fn test() {
    let mut mlp = MultiLayerPerceptron::new(vec![2, 3, 4, 1]);

    let database = vec![(vec![0f64, 0f64], vec![0f64], 1f64)];

    let test_input = vec![0f64, 0f64];

    for i in 1..=50 {
        mlp.backpropagation(database.clone(), 10_000 * i, 0.1 / i as f64);

        let test = mlp.calc(test_input.clone());

        println!();
        println!("TEST : {:?}", test);
        // sleep(Duration::from_millis(50));
    }
}
