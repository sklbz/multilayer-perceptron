use crate::multilayer_perceptron::multilayer_perceptron::*;

#[allow(unused)]
pub fn test() {
    let mut mlp = MultiLayerPerceptron::new(vec![2, 64, 64, 64, 64, 3]);

    let calc = mlp.calc(vec![0.0, 0.0]);

    println!("Calculation: {:?}", calc);

    // mlp.display();

    let database = vec![(vec![0.0, 0.0], vec![1.0, 0.0, 0.0], 1.0)];

    // mlp.train(database);

    mlp.backpropagation(database);
}
