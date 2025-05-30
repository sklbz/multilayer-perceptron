use crate::mlp::multilayer_perceptron::*;

#[allow(unused)]
use std::thread::sleep;
#[allow(unused)]
use std::time::Duration;

#[allow(unused)]
pub fn test() {
    let mut mlp = MultiLayerPerceptron::new(vec![2, 1]);

    let database = vec![
        (vec![0.0, 0.0], vec![0.0], 1.0),
        (vec![0.0, 1.0], vec![0.0], 1.0),
        (vec![1.0, 0.0], vec![0.0], 1.0),
        (vec![1.0, 1.0], vec![0.0], 1.0),
        (vec![-1.0, 0.0], vec![0.0], 1.0),
        (vec![0.0, -1.0], vec![0.0], 1.0),
        (vec![-1.0, -1.0], vec![0.0], 1.0),
        (vec![1.0, -1.0], vec![0.0], 1.0),
        (vec![-1.0, 1.0], vec![0.0], 1.0),
        (vec![0.0, 2.0], vec![0.0], 1.0),
        (vec![2.0, 0.0], vec![0.0], 1.0),
        (vec![2.0, 2.0], vec![0.0], 1.0),
        (vec![-2.0, 0.0], vec![0.0], 1.0),
        (vec![0.0, -2.0], vec![0.0], 1.0),
        (vec![-2.0, -2.0], vec![0.0], 1.0),
        (vec![2.0, -2.0], vec![0.0], 1.0),
        (vec![-2.0, 2.0], vec![0.0], 1.0),
    ];

    let test_input = vec![0.0, 0.0];

    for i in 1..=500 {
        mlp.backpropagation(database.clone(), 10, 0.01);

        let test = mlp.calc(test_input.clone())[0];

        println!();
        println!("TEST {i}: {:?}", test);
        sleep(Duration::from_millis(50));
    }
}
